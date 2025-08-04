import requests
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

from datasets import load_dataset, Dataset
import time
from packing.utils.functions import (
    string_to_function,
    extract_functions,
    extract_imports,
    function_to_string,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
import torch.cuda
import torch.distributed
import gc
from omegaconf import DictConfig, OmegaConf
from accelerate.utils import release_memory
import multiprocessing
import threading
from omegaconf import ListConfig
import logging
from accelerate import Accelerator
import torch.multiprocessing as mp
import os
import signal
import subprocess
import socket
from huggingface_hub import InferenceClient
from concurrent.futures import ThreadPoolExecutor, as_completed

from packing.utils.vllm import dict_to_namespace


def get_full_model_name(cfg):
    def get_name(name):
        if name == "granite":
            model_id = "ibm-granite/granite-3.1-2b-instruct"
        elif name == "llama32":
            model_id = "meta-llama/Llama-3.2-1B-Instruct"
        elif name == "phi":
            model_id = "microsoft/Phi-3.5-mini-instruct"
        else:
            raise ValueError(f"Invalid model name: {name}")
        return model_id

    if isinstance(cfg.model.model_name, (list, ListConfig)):
        return [get_name(name) for name in cfg.model.model_name]
    else:
        return get_name(cfg.model.model_name)


def initialize_single_model(
        cfg,
        full_model_name,
        load_finetuned,
        train=False,
        model_adapter_dir=None,
        gpu_num=0,
        num_outputs_per_prompt=1,
):
    """
    Initialize a single Huggingface model for training or inference.
    """

    if cfg.model_dtype == "float16":
        torch_dtype = torch.float16
    elif cfg.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    if not train:
        init_args = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": {"": gpu_num},
        }
    else:
        init_args = {}

    if cfg.flash_attn:
        init_args["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(full_model_name, **init_args)

    logging.info(f"Loaded model {full_model_name} on GPU {model.device.index}")

    if load_finetuned:
        # Load the finetuned model for inference.
        logging.info(f"Using the finetuned model from {model_adapter_dir}")
        model = AutoModelForCausalLM.from_pretrained(model_adapter_dir, **init_args)
    else:
        # Initialize the Huggingface model for training or inference
        model = AutoModelForCausalLM.from_pretrained(full_model_name, **init_args)
        logging.info("Not using a finetuned model")

    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    # tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    sampling_params = GenerationConfig.from_pretrained(
        full_model_name,
        num_return_sequences=num_outputs_per_prompt,
        max_new_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
        top_k=cfg.model.topk,
        top_p=cfg.model.topp,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )  # , penalty=1.0, repetition_penalty=1.0)

    return model, tokenizer, sampling_params


def initialize_models(cfg, load_finetuned):
    """
    Initialize the inference model, tokenizer, and sampling parameters for the model(s) specified in the config file.
    """

    if cfg.multiple_models:
        assert isinstance(cfg.model.model_name, (list, ListConfig))
        assert isinstance(cfg.full_model_name, (list, ListConfig))
        assert isinstance(cfg.gpu_nums, (list, ListConfig))
        assert len(cfg.model.model_name) == len(cfg.gpu_nums) == len(cfg.full_model_name)

        output_model = []
        output_tokenizer = []
        output_sampling_params = []
        for full_model_name, gpu_num, model_adapter_dir, num_outputs_per_prompt in zip(
                cfg.full_model_name, cfg.gpu_nums, cfg.model_adapter_dir, cfg.num_outputs_per_prompt
        ):
            model, tokenizer, sampling_params = initialize_single_model(
                cfg,
                full_model_name,
                load_finetuned,
                train=False,
                model_adapter_dir=model_adapter_dir,
                gpu_num=gpu_num,
                num_outputs_per_prompt=num_outputs_per_prompt,
            )

            output_model.append(model)
            output_tokenizer.append(tokenizer)
            output_sampling_params.append(sampling_params)
            logging.info(
                f"Cuda memory allocated: {torch.cuda.memory_allocated() // 1024 // 1024}MB"
            )
    else:
        output_model, output_tokenizer, output_sampling_params = initialize_single_model(
            cfg,
            cfg.full_model_name,
            load_finetuned,
            train=False,
            model_adapter_dir=cfg.model_adapter_dir,
            gpu_num=cfg.gpu_nums,
            num_outputs_per_prompt=cfg.num_outputs_per_prompt,
        )

    return output_model, output_tokenizer, output_sampling_params


def generate_on_gpu(chat, model, tokenizer, sampling_params, gpu_num, load_finetuned):
    tokenized_chat = tokenizer.apply_chat_template(
        chat, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(
        f"cuda:{gpu_num}"
    )

    logging.info(f"Tokenization finished, generating on GPU {model.hf_device_map}...")
    model.eval()

    logging.info(f"Generating with temperature: {sampling_params.temperature}")

    outputs = model.generate(
        tokenized_chat, generation_config=sampling_params, pad_token_id=tokenizer.eos_token_id
    )

    # Use list comprehension to process each output
    outputs_text_init = [
        tokenizer.decode(output[len(tokenized_chat[0]):], skip_special_tokens=True)
        for output in outputs
    ]

    extracted_functions_init = [extract_functions(output) for output in outputs_text_init]
    extracted_imports_init = [extract_imports(output) for output in outputs_text_init]

    # Filter out texts, functions, and imports where functions are extracted successfully
    extracted_functions = [func for func in extracted_functions_init if func]
    extracted_imports = [
        imports for i, imports in enumerate(extracted_imports_init) if extracted_functions_init[i]
    ]
    filtered_outputs_text = [
        output for i, output in enumerate(outputs_text_init) if extracted_functions_init[i]
    ]

    assert (
            len(extracted_functions)
            == len(extracted_imports)
            == len(filtered_outputs_text)
    )

    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
    )


def get_outputs_from_single_model(cfg, chat, model, tokenizer, sampling_params, load_finetuned):
    time_start = time.time()

    (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
    ) = generate_on_gpu(chat, model, tokenizer, sampling_params, cfg.gpu_nums, load_finetuned)

    llm_generation_time = int(time.time() - time_start)
    logging.info(f"Time taken for LLM generation: {llm_generation_time} seconds")

    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
        llm_generation_time,
    )


def get_outputs_from_multiple_models(
        cfg, chat, models_list, tokenizers_list, sampling_params_list, load_finetuned
):
    # Track the time taken for generation
    time_start = time.time()

    model_count = len(models_list)

    # List to store outputs
    outputs_text_collect = [None] * model_count
    functions_collect = [None] * model_count
    imports_collect = [None] * model_count
    llm_generation_time_collect = [None] * model_count

    # Function to run generate
    def single_thread_generation(chat, model, tokenizer, sampling_params, gpu_num, idx):

        (
            extracted_functions,
            extracted_imports,
            outputs_text,
        ) = generate_on_gpu(chat, model, tokenizer, sampling_params, gpu_num, load_finetuned)

        llm_generation_time = int(time.time() - time_start)
        logging.info(f"Time taken for LLM generation: {llm_generation_time} seconds")

        outputs_text_collect[idx] = outputs_text
        functions_collect[idx] = extracted_functions
        imports_collect[idx] = extracted_imports
        llm_generation_time_collect[idx] = llm_generation_time

    threads = []
    # Create threads for parallel execution
    for model, tokenizer, sampling_params, gpu_num, idx in zip(
            models_list, tokenizers_list, sampling_params_list, cfg.gpu_nums, range(model_count)
    ):
        thread = threading.Thread(
            target=single_thread_generation,
            args=(chat, model, tokenizer, sampling_params, gpu_num, idx),
        )
        threads.append(thread)
    # Start threads
    for thread in threads:
        thread.start()

    # Wait for threads to finish
    for thread in threads:
        thread.join()

    # Assert there are no None values in the lists
    assert None not in outputs_text_collect
    assert None not in functions_collect
    assert None not in imports_collect
    assert None not in llm_generation_time_collect

    # Assert that the lengths of the lists are equal
    assert (
            len(outputs_text_collect)
            == len(functions_collect)
            == len(imports_collect)
            == len(llm_generation_time_collect)
    )

    # Unflatten the outputs
    extracted_functions = [func for sublist in functions_collect for func in sublist]
    extracted_imports = [imports for sublist in imports_collect for imports in sublist]
    filtered_outputs_text = [output for sublist in outputs_text_collect for output in sublist]
    llm_generation_time = max(llm_generation_time_collect)

    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
        llm_generation_time,
    )


def make_tgi_request(cfg, chat, port):
    client = InferenceClient(base_url=f"http://localhost:{port}/v1/")
    response = client.chat.completions.create(
        model="tgi",
        messages=chat,
        stream=False,
        max_tokens=cfg.model.max_tokens,
        temperature=cfg.model.temperature,
        top_p=cfg.model.topp,
    )
    return response


def make_vllm_request(cfg, chat, load_finetuned, port):
    """
    Send chat request to a vLLM server running on localhost:<port>.

    Args:
        cfg: A config object that contains, at minimum, `full_model_name`.
        chat: A list of message dicts, e.g. [{"role": "user", "content": "Hello!"}, ...]
        load_finetuned: A boolean specifying whether we are using a fine-tuned model or not
        port: The integer port where the vLLM server is listening.

    Returns:
        The JSON response from the vLLM server.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": cfg.full_model_name if not load_finetuned else cfg.model_adapter_dir,
        "messages": chat,
        "max_tokens": cfg.model.max_tokens,
        "temperature": cfg.model.temperature,
        "top_p": cfg.model.topp,
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()  # Raises an HTTPError if the response was unsuccessful
    return dict_to_namespace(response.json())


def generate_from_server(cfg, chat, flag_load_finetuned, server_port, num_outputs_per_prompt):
    outputs = []
    with ThreadPoolExecutor(max_workers=num_outputs_per_prompt) as executor:
        futures = [
            executor.submit(make_tgi_request, cfg, chat, server_port) if cfg.use_tgi else
            executor.submit(make_vllm_request, cfg, chat, flag_load_finetuned, server_port)
            for _ in range(num_outputs_per_prompt)
        ]

        for idx, future in enumerate(as_completed(futures)):
            try:
                outputs.append(future.result())
            except Exception as e:
                print(f"Request failed: {e}")

    outputs = [output.choices[0].message.content for output in outputs]

    extracted_functions_init = [extract_functions(output) for output in outputs]
    extracted_imports_init = [extract_imports(output) for output in outputs]

    # Filter out texts, functions, and imports where functions are extracted successfully
    extracted_functions = [func for func in extracted_functions_init if func]
    extracted_imports = [
        imports for i, imports in enumerate(extracted_imports_init) if extracted_functions_init[i]
    ]
    filtered_outputs_text = [
        output for i, output in enumerate(outputs) if extracted_functions_init[i]
    ]

    assert (
            len(extracted_functions)
            == len(extracted_imports)
            == len(filtered_outputs_text)
    )

    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
    )


def get_outputs_from_single_model_server(cfg, chat, server_ports, flag_load_finetuned):
    time_start = time.time()

    (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
    ) = generate_from_server(
        cfg, chat, flag_load_finetuned, server_ports[0], num_outputs_per_prompt=cfg.num_outputs_per_prompt
    )

    llm_generation_time = int(time.time() - time_start)
    logging.info(f"Time taken for LLM generation: {llm_generation_time} seconds")

    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
        llm_generation_time,
    )


def get_outputs_from_multiple_models_server(cfg, chat, server_ports, flag_load_finetuned):
    # Track the time taken for generation
    time_start = time.time()

    model_count = len(cfg.full_model_name)

    # List to store outputs
    outputs_text_collect = [None] * model_count
    functions_collect = [None] * model_count
    imports_collect = [None] * model_count
    llm_generation_time_collect = [None] * model_count

    chats = [chat] * model_count

    with ThreadPoolExecutor(max_workers=model_count) as executor:

        futures = [
            executor.submit(generate_from_server, cfg, chat, flag_load_finetuned, port, num_outputs_per_prompt)
            for chat, port, num_outputs_per_prompt in zip(
                chats, server_ports, cfg.num_outputs_per_prompt
            )
        ]

        for idx, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                (
                    extracted_functions,
                    extracted_imports,
                    filtered_outputs_text,
                ) = result
                outputs_text_collect[idx] = filtered_outputs_text
                functions_collect[idx] = extracted_functions
                imports_collect[idx] = extracted_imports
                # WARNING: This is not the actual generation time as calculating that with multithreading is not straightforward
                llm_generation_time = int(time.time() - time_start)
                llm_generation_time_collect[idx] = llm_generation_time
                logging.info(f"Time taken for LLM generation: {llm_generation_time} seconds")
            except Exception as e:
                print(f"Request failed: {e}")

    # Assert there are no None values in the lists
    assert None not in outputs_text_collect
    assert None not in functions_collect
    assert None not in imports_collect
    assert None not in llm_generation_time_collect

    # Assert that the lengths of the lists are equal
    assert (len(outputs_text_collect)
            == len(functions_collect)
            == len(imports_collect)
            == len(llm_generation_time_collect)
    )

    # Unflatten the outputs
    extracted_functions = [func for sublist in functions_collect for func in sublist]
    extracted_imports = [imports for sublist in imports_collect for imports in sublist]
    filtered_outputs_text = [output for sublist in outputs_text_collect for output in sublist]
    llm_generation_time = max(llm_generation_time_collect)

    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
        llm_generation_time,
    )


def get_outputs(cfg, chat, *args):
    """
    Routs the generation request to the appropriate function based on the number of models.
    """

    server_ports, model, tokenizer, sampling_params, flag_load_finetuned = None, None, None, None, None
    if cfg.use_tgi or cfg.use_vllm:
        server_ports = args[0][0]
        flag_load_finetuned = args[1]
    else:
        model, tokenizer, sampling_params, flag_load_finetuned = args

    if cfg.multiple_models:
        (
            extracted_functions,
            extracted_imports,
            filtered_outputs_text,
            llm_generation_time,
        ) = get_outputs_from_multiple_models_server(cfg, chat, server_ports,
                                                    flag_load_finetuned) if cfg.use_tgi or cfg.use_vllm \
            else get_outputs_from_multiple_models(cfg, chat, model, tokenizer, sampling_params, flag_load_finetuned)
    else:
        (
            extracted_functions,
            extracted_imports,
            filtered_outputs_text,
            llm_generation_time,
        ) = get_outputs_from_single_model_server(cfg, chat, server_ports,
                                                 flag_load_finetuned) if cfg.use_tgi or cfg.use_vllm \
            else get_outputs_from_single_model(cfg, chat, model, tokenizer, sampling_params, flag_load_finetuned)
    return (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
        llm_generation_time,
    )


def initialize_models_server(cfg, load_finetuned, use_vllm=False):
    if cfg.multiple_models:
        assert isinstance(cfg.model.model_name, (list, ListConfig))
        assert isinstance(cfg.full_model_name, (list, ListConfig))
        assert isinstance(cfg.gpu_nums, (list, ListConfig))
        assert len(cfg.model.model_name) == len(cfg.gpu_nums) == len(cfg.full_model_name)

        server_pids = []
        model_ids = []
        ports = [8080 + i for i in range(len(cfg.model.model_name))]

        for full_model_name, gpu_num, model_adapter_dir, port in zip(
                cfg.full_model_name, cfg.gpu_nums, cfg.model_adapter_dir, ports
        ):
            model_id = full_model_name if not load_finetuned else model_adapter_dir

            if not use_vllm:
                pid = start_tgi_server(model_id, gpu_num, port)
            else:
                pid = start_vllm_server(model_id, gpu_num, port)
            server_pids.append(pid)
            model_ids.append(model_id)
            # wait to avoid crashing by initializing consecutive servers too quickly
            # time.sleep(20)
            if not use_vllm:
                wait_for_tgi_server(port, model_id)
            else:
                wait_for_vllm_server(port, model_id)
            # delete the ./out folder that gets created by tgi
            subprocess.run(["rm", "-rf", "./out"])

            logging.info(
                f"Cuda memory allocated: {torch.cuda.memory_allocated() // 1024 // 1024}MB"
            )
    else:
        ports = [8080 + cfg.gpu_nums]  # Hacky but this will make sure we have unique server ports for each model
        model_id = cfg.full_model_name if not load_finetuned else cfg.model_adapter_dir

        if not use_vllm:
            pid = start_tgi_server(model_id, cfg.gpu_nums, ports[0])
        else:
            pid = start_vllm_server(model_id, cfg.gpu_nums, ports[0])

        server_pids = [pid]
        model_ids = [model_id]

    # waiting for all of the servers to start
    logging.info(f"Waiting for all servers to start... Server PIDs: {server_pids}")
    for port, model_id in zip(ports, model_ids):
        try:
            if not use_vllm:
                wait_for_tgi_server(port, model_id)
            else:
                wait_for_vllm_server(port, model_id)
        except TimeoutError as e:
            logging.error(e)

    return server_pids, ports


def start_tgi_server(model_id, gpu_num, port):
    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
    logging.info(f"Starting TGI server for model {model_id} on GPU {gpu_num} at port {port}")
    # command = ["text-generation-launcher", "--model-id", model_id, "--tokenizer-config-path", tokenizer_path, "--port", str(port)]
    command = ["text-generation-launcher", "--model-id", model_id, "--port", str(port)]
    process = subprocess.Popen(command, env=env_vars)
    pid = process.pid
    return pid


def start_vllm_server(model_id, gpu_num, port):
    """
    Starts a vLLM server with OpenAI-compatible API, assigning the specified GPU.
    Refer to vLLM docs for additional flags/options:
    https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve
    """
    env_vars = os.environ.copy()
    # Limit the server process to the specified GPU
    env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

    logging.info(f"Starting vLLM server for model {model_id} on GPU {gpu_num} at port {port}")

    # Basic command: serve the model on the given port and host
    command = [
        "vllm",
        "serve",
        model_id,
        "--port", str(port),
        "--disable-log-requests"
    ]

    # If you need additional arguments, append them here. For example:
    # command += ["--tensor-parallel-size", "1"]
    # command += ["--max-num-batches", "4"]
    # etc.

    process = subprocess.Popen(command, env=env_vars)
    return process.pid


def kill_process_with_pid_and_wait(pid):
    os.kill(pid, signal.SIGTERM)
    # Wait for the process to terminate so we're sure memory is free for training
    while True:
        try:
            # Check the process state
            pid, _ = os.waitpid(pid, os.WNOHANG)
            if pid == 0:
                # Process is still running, sleep for a short duration
                time.sleep(0.1)
            else:
                # Process has terminated
                logging.info(f"Server process {pid} has terminated successfully.")
                break
        except ChildProcessError:
            # No child processes, so it's already terminated
            logging.info(f"Server process {pid} has terminated successfully.")
            break


def wait_for_tgi_server(port, model_id, timeout=1000):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Attempt to connect to the server
            with socket.create_connection(("localhost", port), timeout=1):
                logging.info(f"Server is ready at port {port}. Serving {model_id}.")
                return
        except (ConnectionRefusedError, socket.timeout):
            # Server is not ready yet
            time.sleep(1)
    raise TimeoutError(
        f"Server at port {port} did not start within {timeout} seconds for model {model_id}."
    )


def wait_for_vllm_server(port, model_id, timeout=1000):
    """
    Waits for the vLLM server to become ready at http://localhost:<port>/v1/chat/completions
    by sending minimal requests. Retries until timeout.

    Args:
        model_id (str): The full model name or directory
        port (int): The port number on which vLLM is expected to be running.
        timeout (int): Maximum number of seconds to wait before giving up.

    Raises:
        TimeoutError: If the server is not ready before the timeout.
    """
    start_time = time.time()
    url = f"http://localhost:{port}/v1/chat/completions"

    # A minimal request payload. vLLM typically requires `model` and `messages`.
    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
    }

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            raise TimeoutError(f"vLLM server not ready after {timeout} seconds on port {port}.")

        try:
            resp = requests.post(url, json=payload, timeout=1)
            # If it returns a 200 OK, we consider the server ready
            if resp.status_code == 200:
                logging.info(f"Server is ready at port {port}. Serving {model_id}.")
                return  # Server is ready
        except requests.exceptions.RequestException:
            # Connection errors, timeouts, etc. => server not ready yet
            pass

        time.sleep(1)


def kill_process(pid):
    try:
        subprocess.run(["kill", "-9", str(pid)], check=True)
        logging.info(f"Process with PID {pid} killed successfully (Port should be released).")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to kill process with PID {pid}: {e}")


def clean_up_gpu_mem():
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.set_device(device_idx)
        logging.info(
            f"  --> Device {device_idx} memory before cleanup: "
            f"{torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
        )
        # for _ in range(2):
        torch.cuda.empty_cache()
        # time.sleep(1)

        # for _ in range(2):
        gc.collect()
        #    time.sleep(1)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        release_memory()
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(
            f"  --> Device {device_idx} memory after cleanup: "
            f"{torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
        )
    # Optional: reset to primary device
    torch.cuda.set_device(0)


def delete_sampling_model(cfg, model, tokenizer, sampling_params):
    if cfg.multiple_models:
        for single_model in model:
            single_model = single_model.cpu()
            del single_model
        for single_tokenizer in tokenizer:
            del single_tokenizer
    else:
        model = model.cpu()
        del model
        del tokenizer
    clean_up_gpu_mem()
    del sampling_params
