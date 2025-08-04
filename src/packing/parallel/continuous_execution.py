from typing import Union, List

import psutil
import logging
import queue
import traceback

from packing.funsearch.programs_database import ProgramsDatabase
from packing.parallel.stoppable_task import StoppableTask
import wandb
from packing.logging.function_class import FunctionClass
import copy
import numpy as np
import time
from packing.model.model import get_outputs
import multiprocessing
from packing.utils.functions import separate_imports_from_func
import math
import os
import json

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import tensorflow as tf

    HAS_TF = True
except ImportError:
    HAS_TF = False


def is_memory_error(err):
    """Checks if an exception is a memory-related error."""
    memory_errors = [MemoryError, np.core._exceptions._ArrayMemoryError, psutil.Error]

    if HAS_TORCH:
        memory_errors.append(torch.cuda.OutOfMemoryError)

    if HAS_CUPY:
        memory_errors.append(cupy.cuda.memory.OutOfMemoryError)

    if HAS_TF:
        memory_errors.append(tf.errors.ResourceExhaustedError)

    return isinstance(err, tuple(memory_errors)) or (
            isinstance(err, OSError) and "Cannot allocate memory" in str(err)
    ) or (
            "numpy.core._exceptions._ArrayMemoryError" in str(err)
    )


def generate_functions(
        cfg,
        chat,
        prompt_scores,
        island_id_prompt,
        temperature,
        probabilities,
        prompt_num,
        flag_load_finetuned,
        server_ports=None,
        model=None,
        tokenizer=None,
        sampling_params=None,
):
    """
    Generate functions for a given prompt and wrap them in FunctionClass object.
    """

    logging.info("\n\n")
    logging.info(f"Prompt scores: {prompt_scores}")
    logging.info(f"Island id {island_id_prompt}")
    logging.info("\n\n")
    if cfg.use_tgi or cfg.use_vllm:
        args = (server_ports, flag_load_finetuned)
    else:
        args = (model, tokenizer, sampling_params, flag_load_finetuned)
    (
        extracted_functions,
        extracted_imports,
        filtered_outputs_text,
        llm_generation_time,
    ) = get_outputs(cfg, chat, *args)

    function_classes = []
    failed_function_classes = []
    num_func_generated = 0
    for function_str, imports_str, output_text in zip(
            extracted_functions,
            extracted_imports,
            filtered_outputs_text,
    ):
        num_func_generated += 1
        chat_copy = copy.deepcopy(chat)
        chat_copy.append({"role": "assistant", "content": output_text})

        # Every generated function is wrapped in a FunctionClass object, which tracks everything associated with it
        function_class = FunctionClass()
        function_class.function_str = function_str
        function_class.imports_str = imports_str
        function_class.prompt.island_id = island_id_prompt
        function_class.prompt.min_prompt_score = min(prompt_scores)
        function_class.prompt.max_prompt_score = max(prompt_scores)
        function_class.prompt.num_examples_prompt = len(prompt_scores)
        function_class.prompt.temperature = temperature
        function_class.prompt.max_prob = np.max(probabilities)
        function_class.prompt.min_prob = np.min(probabilities)
        function_class.chat = chat_copy
        function_class.function_num = num_func_generated
        function_class.prompt_num = prompt_num

        if function_str.find(cfg.function_str_to_extract) == -1:
            logging.info(
                f"Function {cfg.function_str_to_extract} not found in the extracted function."
            )
            function_class.fail_flag = 1
            function_class.score = cfg.task.failed_score
            function_class.true_score = cfg.task.failed_score
            failed_function_classes.append(function_class)
        else:
            function_classes.append(function_class)

    logging.info(
        f"Number of outputs with at least one valid function definition: {len(function_classes)}"
    )

    return function_classes, failed_function_classes, llm_generation_time, num_func_generated


def Producer(
        cfg,
        # server_ports,
        function_classes_to_eval,
        failed_function_classes_global,
        chats_cont,
        prompt_scores_cont,
        island_id_prompt_cont,
        temperatures_cont,
        probabilities_cont,
        prompt_nums_cont,
        multiprocess_running_dict,
        termination_event,
        flag_load_finetuned,
        *args
):
    """
    Producer process that generates functions and puts them in the function_classes_to_eval queue.
    """
    try:
        for (
                chat,
                prompt_scores,
                island_id_prompt,
                temperature,
                probabilities,
                prompt_num,
        ) in zip(
            chats_cont,
            prompt_scores_cont,
            island_id_prompt_cont,
            temperatures_cont,
            probabilities_cont,
            prompt_nums_cont,
        ):
            if cfg.use_tgi or cfg.use_vllm:
                server_ports = args
                (
                    function_classes,
                    failed_function_classes,
                    llm_time,
                    num_generated,
                ) = generate_functions(
                    cfg,
                    chat,
                    prompt_scores,
                    island_id_prompt,
                    temperature,
                    probabilities,
                    prompt_num,
                    flag_load_finetuned,
                    server_ports=server_ports,
                )
            else:
                model, tokenizer, sampling_params = args
                (
                    function_classes,
                    failed_function_classes,
                    llm_time,
                    num_generated,
                ) = generate_functions(
                    cfg,
                    chat,
                    prompt_scores,
                    island_id_prompt,
                    temperature,
                    probabilities,
                    prompt_num,
                    flag_load_finetuned,
                    model=model,
                    tokenizer=tokenizer,
                    sampling_params=sampling_params,
                )
            multiprocess_running_dict["llm_generation_time"].append(llm_time)
            multiprocess_running_dict["num_func_generated"] += num_generated
            logging.info(f"LLM generation time: {llm_time}")

            for func_class in function_classes:
                function_classes_to_eval.put(func_class)
            logging.info(
                f"Producer: Generated {len(function_classes)} functions for prompt {prompt_num}"
            )

            for failed_func_class in failed_function_classes:
                failed_function_classes_global.append(failed_func_class)
            logging.info(
                f"Producer: Generated {len(failed_function_classes)} failed functions for prompt {prompt_num}"
            )

    except Exception as e:
        logging.error(f"Producer encountered an error: {e}\n{traceback.format_exc()}")
        termination_event.set()
        return  # Exit the function

    finally:
        # Send termination signal to all consumers
        for _ in range(cfg.num_workers):
            function_classes_to_eval.put(None)
        logging.info("Producer: Sent stopping signals to consumers")


def Consumer(
        cfg,
        evaluate_func,
        function_classes_to_eval,
        passed_function_classes_global,
        failed_function_classes_global,
        worker_id,
        termination_event,
        input_struct,
        round_num,
        multiprocess_running_dict,
):
    """
    Consumer process that evaluates functions from thefunction_classes_to_eval queue.
    """
    try:
        while not termination_event.is_set():
            try:
                # Get next function with timeout to check termination periodically
                logging.info(f"[Worker {worker_id}] Waiting for function...")
                try:
                    function_class = function_classes_to_eval.get(timeout=10)
                except queue.Empty:
                    continue

                # Check for sentinel value
                if function_class is None:
                    logging.info(f"[Worker {worker_id}] Received stopping None; exiting")
                    break

                # Log queue size if available
                try:
                    queue_size = function_classes_to_eval.qsize()
                    logging.info(f"[Worker {worker_id}] Queue size is {queue_size}")
                except NotImplementedError:
                    pass

                # Evaluate the function
                function_class.eval.idx_process = worker_id

                try:
                    with StoppableTask(
                            cfg,
                            evaluate_func,
                            copy.deepcopy(input_struct),
                            function_class,
                            worker_id,
                            timeout=cfg.task.timeout_period,
                    ) as task:
                        task.run()

                        while (
                                task.is_alive()
                        ):  # return self.process is not None and self.process.is_alive()
                            finished = task.join(1)
                            if finished:
                                logging.info(f"[Worker {worker_id}] Task finished: {finished}")
                                break

                        if not task.is_timed_out():
                            # No timeout kill happened
                            result_tuple = task.get_result()
                            if result_tuple is not None:
                                task_id, result_class = result_tuple
                                # Distinguish normal results vs. "Exception: ..." strings
                                if isinstance(result_class, FunctionClass):
                                    if result_class.fail_flag == 0:
                                        passed_function_classes_global.append(result_class)
                                        logging.info(
                                            f"[Worker {task_id}] Completed. score = {result_class.score}, true_score = {result_class.true_score}"
                                        )
                                    elif result_class.fail_flag == 1:
                                        # Mark as failed
                                        assert result_class.score == cfg.task.failed_score
                                        assert result_class.true_score == cfg.task.failed_score
                                        failed_function_classes_global.append(result_class)
                                    else:
                                        logging.error("Invalid fail flag")
                                        raise ValueError("Invalid fail flag")
                                elif isinstance(result_class, str) and "Exception:" in result_class:
                                    # An exception was raised by the evaluate function
                                    logging.error(
                                        f"[Worker {task_id}] Evaluate function raised an exception: {result_class}"
                                    )
                                    raise ValueError(
                                        f"[Worker {task_id}] Evaluate function exception: {result_class}"
                                    )
                                else:
                                    # Unexpected data type
                                    logging.error(
                                        f"[Worker {task_id}] Returned an unexpected result: {result_class}"
                                    )
                                    raise ValueError(
                                        f"[Worker {task_id}] Unexpected result: {result_class}"
                                    )
                            else:
                                # Result queue empty
                                with task._process_lock:
                                    exit_code = task.process.exitcode if task.process else None

                                msg = (
                                    f"[Worker {worker_id}] Task {task.task_id} completed "
                                    f"with NO result in the queue."
                                )
                                if exit_code is not None:
                                    msg += f" Subprocess exit code={exit_code}."
                                    if exit_code != 0:
                                        msg += " Non-zero exit => crash or forced termination."
                                else:
                                    msg += " Could not read exit code (process missing?)."

                                logging.error(msg)
                        else:
                            # We timed out and killed the task
                            logging.info(f"[Worker {task.task_id}] Timed out, marking as failed.")
                            failed_function_class = task.function_class
                            failed_function_class.fail_flag = 1
                            failed_function_class.score = cfg.task.failed_score
                            failed_function_class.true_score = cfg.task.failed_score
                            failed_function_classes_global.append(failed_function_class)
                            logging.info(
                                f"Function that timed out: {failed_function_class.function_str}"
                            )

                except Exception as task_error:
                    logging.error(
                        f"[Worker {worker_id}] Task error: {task_error}\n{traceback.format_exc()}"
                    )

                    if not is_memory_error(task_error):
                        raise  # Re-raise to be caught by outer try/except

            except Exception as e:
                # Catch any other exceptions, set termination_event so other consumers stop
                logging.error(
                    f"[Worker {worker_id}] Consumer loop error: {e}\n{traceback.format_exc()}"
                )
                if not termination_event.is_set():
                    termination_event.set()
                break

    except Exception as e:
        logging.error(f"[Worker {worker_id}] Consumer process error: {e}")
        termination_event.set()
        return

    finally:
        logging.info(f"[Worker {worker_id}] Shutting down consumer process")
        # Additional cleanup:
        import os

        logging.info(f"[Worker {worker_id}] (pid={os.getpid()}) in finally block.")

        # If there's a local variable 'task' from StoppableTask, ensure it's fully stopped:
        if "task" in locals():
            if task.is_alive():
                logging.info(f"[Worker {worker_id}] Forcing a final stop()")
                task.stop()

        # Optionally remove references to large objects to help GC:
        try:
            del function_class
            del input_struct
        except NameError:
            pass

        import gc

        gc.collect()
        logging.info(f"[Worker {worker_id}] cleanup complete.")


def consumers_finish_and_cleanup(
        cfg,
        consumers,
        function_classes_to_eval,
):
    """
    Wait for all consumers to finish, collect the results, and clean up.
    """

    # Wait for all consumers to finish
    logging.info("WAITING FOR CONSUMERS TO FINISH")
    try:
        for worker_id, p in enumerate(consumers):
            try:
                logging.info(
                    f"Waiting for consumer process {p.pid}, worker id {worker_id} to finish"
                )
                queue_size = function_classes_to_eval.qsize()
                timeout_time = math.ceil(
                    (queue_size + 1) / (cfg.num_workers / 2)) * cfg.task.timeout_period + cfg.task.timeout_period
                logging.info(f"Queue size: {queue_size}, timeout time: {timeout_time}")

                p.join(timeout=timeout_time)
                logging.info(f"Consumer process {p.pid} exited")
            except Exception as e:
                logging.error(f"Error during shutdown of process {p.pid}: {e}")
                raise ValueError(f"Error during shutdown of process {p.pid}: {e}")
    except Exception as e:
        logging.error(f"Error during consumer shutdown: {e}")
        raise ValueError(f"Error during shutdown {e}")


def EvalProducer(
        cfg,
        function_classes_to_eval,
        multiprocess_running_dict,
        termination_event,
        programdatabase: Union[ProgramsDatabase, List[FunctionClass]],
):
    """
    Producer process that takes functions from the programdatabase and evaluates them on the new input
    """
    try:
        if type(programdatabase) == ProgramsDatabase:
            for island in programdatabase._islands:
                # for cluster in island._clusters:
                for cluster_key in island._clusters.keys():
                    for function_str_with_imports in island._clusters[cluster_key]._programs:
                        # logging.info(f"Function string with imports: {function_str_with_imports}")
                        # Separate the imports from the function string
                        imports_str, function_str = separate_imports_from_func(
                            function_str_with_imports
                        )
                        function_class = FunctionClass()
                        function_class.function_str = function_str
                        function_class.imports_str = imports_str
                        function_class.original_score = island._clusters[cluster_key]._score
                        # function_class.prompt.island_id = island_id_prompt
                        function_classes_to_eval.put(function_class)

                        multiprocess_running_dict["num_func_taken_from_programdb"] += 1
                        # multiprocess_running_dict["llm_generation_time"].append(0)
        else:
            for existing_function_class in programdatabase:
                function_class = FunctionClass()
                function_class.function_str = existing_function_class.function_str
                function_class.imports_str = existing_function_class.imports_str
                function_class.original_score = existing_function_class.original_score
                # function_class.prompt.island_id = island_id_prompt
                function_classes_to_eval.put(function_class)

                multiprocess_running_dict["num_func_taken_from_programdb"] += 1

    except Exception as e:
        logging.error(f"EvalProducer encountered an error: {e}\n{traceback.format_exc()}")
        termination_event.set()
        return  # Exit the function

    finally:
        # Send termination signal to all consumers
        for _ in range(cfg.num_workers):
            function_classes_to_eval.put(None)
        logging.info("Producer: Sent stopping signals to consumers")


def evaluate_on_evalset(
        cfg, running_dict, round_num, evaluate_func, programdatabase, evalset, generate_input
):
    eval_wandb_dict = {}
    # Time the evaluation
    start_time = time.time()
    assert type(evalset) == str

    logging.info(f"Starting with evaluating on the {evalset}")
    logging.info(f"Round number: {round_num}")

    # Get new input to evaluate on
    input_struct = generate_input(cfg.task, evalset)

    # Create a queue for functions, lists for failed and passed functions and a dict for running stats
    function_classes_to_eval = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    failed_function_classes_global = manager.list()
    passed_function_classes_global = manager.list()

    multiprocess_running_dict = manager.dict()
    multiprocess_running_dict["num_func_taken_from_programdb"] = 0

    # Create a termination event, if set we know there was an error and we should stop
    termination_event = multiprocessing.Event()

    # Start the consumer processes
    logging.info("-" * 10)
    logging.info("STARTING CONSUMERS")
    consumers = []
    for worker_id in range(cfg.num_workers):
        p = multiprocessing.Process(
            target=Consumer,
            args=(
                cfg,
                evaluate_func,
                function_classes_to_eval,
                passed_function_classes_global,
                failed_function_classes_global,
                worker_id,
                termination_event,
                input_struct,
                round_num,
                multiprocess_running_dict,
            ),
        )
        consumers.append(p)
        p.start()

    # Start the producer thread
    logging.info("-" * 10)
    logging.info(f"STARTING THE EVAL PRODUCER on evalset {evalset}")

    EvalProducer(
        cfg,
        function_classes_to_eval,
        multiprocess_running_dict,
        termination_event,
        programdatabase,
    )
    logging.info(
        f"Producer finished, put {multiprocess_running_dict['num_func_taken_from_programdb']} functions in the queue."
    )

    # Wait for the consumers to finish and clean up
    consumers_finish_and_cleanup(
        cfg,
        consumers,
        function_classes_to_eval,
    )

    # Collect the results, will be registered at the end of the round
    passed_function_classes = copy.deepcopy(list(passed_function_classes_global))
    failed_function_classes = copy.deepcopy(list(failed_function_classes_global))

    logging.info("###################")
    logging.info(f"Num of passed functions on {evalset} set: {len(passed_function_classes)}")
    logging.info(f"Num of failed functions on {evalset} set: {len(failed_function_classes)}")
    logging.info("###################")

    # Clean up
    passed_function_classes_global[:] = []
    failed_function_classes_global[:] = []

    multiprocess_running_dict = dict(multiprocess_running_dict)
    eval_wandb_dict[f"{evalset}/num_func_taken_from_programdb"] = multiprocess_running_dict[
        "num_func_taken_from_programdb"
    ]

    manager.shutdown()
    del passed_function_classes_global
    del failed_function_classes_global
    del multiprocess_running_dict
    del manager

    if termination_event.is_set():
        raise RuntimeError("Termination event was set, there was an error.")

    eval_wandb_dict[f"{evalset}/num_func_passed_on_eval"] = len(passed_function_classes)
    eval_wandb_dict[f"{evalset}/num_func_failed_on_eval"] = len(failed_function_classes)

    # Process the passed_function_classes
    all_scores = []
    train_scores = []
    eval_scores = []
    # all_true_scores = []
    for function_class in passed_function_classes:
        train_scores.append(function_class.original_score)
        eval_scores.append(function_class.true_score)
        all_scores.append(function_class.score)
        # all_true_scores.append(function_class.true_score)

    # Get overall best 10 and 50 scores
    best_10_scores = np.sort(all_scores)[::-1][:10]
    best_10_scores_avg_overall = np.mean(best_10_scores)

    best_50_scores = np.sort(all_scores)[::-1][:50]
    best_50_scores_avg_overall = np.mean(best_50_scores)

    eval_wandb_dict[f"{evalset}/Best score overall"] = np.max(all_scores)
    eval_wandb_dict[f"{evalset}/Best 10 scores overall"] = best_10_scores_avg_overall
    eval_wandb_dict[f"{evalset}/Best 50 scores overall"] = best_50_scores_avg_overall

    eval_wandb_dict[f"{evalset}/Number of unique scores"] = len(set(all_scores))
    eval_wandb_dict[f"{evalset}/Score threshold of best 1% of programs"] = np.percentile(
        all_scores, 99
    )
    eval_wandb_dict[f"{evalset}/Score threshold of best 10% of programs"] = np.percentile(
        all_scores, 90
    )
    eval_wandb_dict[f"{evalset}/Score threshold of best 20% of programs"] = np.percentile(
        all_scores, 80
    )
    eval_wandb_dict[f"{evalset}/Score threshold of best 40% of programs"] = np.percentile(
        all_scores, 60
    )

    end_time = time.time()
    eval_wandb_dict[f"{evalset}/time_taken_to_evaluate"] = end_time - start_time
    eval_wandb_dict[f"round_num"] = round_num
    wandb.log(eval_wandb_dict)

    metrics = {}
    metrics["round_num"] = round_num
    metrics["evalset"] = evalset
    metrics["all_scores"] = all_scores
    metrics["train_eval_scores"] = (train_scores, eval_scores)
    metrics["best_overall_score"] = np.max(all_scores)
    metrics["best_10_scores_avg_overall"] = best_10_scores_avg_overall
    metrics["best_50_scores_avg_overall"] = best_50_scores_avg_overall
    metrics["num_unique_scores"] = len(set(all_scores))
    metrics["score_threshold_best_1_percent"] = np.percentile(all_scores, 99)
    metrics["score_threshold_best_10_percent"] = np.percentile(all_scores, 90)
    metrics["score_threshold_best_20_percent"] = np.percentile(all_scores, 80)
    metrics["score_threshold_best_40_percent"] = np.percentile(all_scores, 60)
    metrics["time_taken_to_evaluate"] = end_time - start_time
    if not os.path.exists(f"{cfg.logs_dir}/metrics"):
        os.makedirs(f"{cfg.logs_dir}/metrics")
    write_to_json(f"{cfg.logs_dir}/metrics/metrics_{evalset}_round_{round_num}.json", metrics)
    # return running_dict


def evaluate_on_dataset(
        cfg,
        running_dict,
        round_num,
        evaluate_func,
        programdatabase,
        dataset,
        generate_input,
):
    eval_wandb_dict = {}
    # Time the evaluation
    start_time = time.time()
    assert type(dataset) == str

    # Get new input to evaluate on
    input_struct = generate_input(cfg.task, dataset)

    # Create a queue for functions, lists for failed and passed functions and a dict for running stats
    function_classes_to_eval = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    failed_function_classes_global = manager.list()
    passed_function_classes_global = manager.list()

    multiprocess_running_dict = manager.dict()
    multiprocess_running_dict["num_func_taken_from_programdb"] = 0

    # Create a termination event, if set we know there was an error and we should stop
    termination_event = multiprocessing.Event()

    # Start the consumer processes
    logging.info("-" * 10)
    logging.info("STARTING CONSUMERS")
    consumers = []
    for worker_id in range(cfg.num_workers):
        p = multiprocessing.Process(
            target=Consumer,
            args=(
                cfg,
                evaluate_func,
                function_classes_to_eval,
                passed_function_classes_global,
                failed_function_classes_global,
                worker_id,
                termination_event,
                input_struct,
                round_num,
                multiprocess_running_dict,
            ),
        )
        consumers.append(p)
        p.start()

    # Start the producer thread
    logging.info("-" * 10)
    logging.info(f"STARTING THE EVAL PRODUCER on dataset {dataset}")

    EvalProducer(
        cfg,
        function_classes_to_eval,
        multiprocess_running_dict,
        termination_event,
        programdatabase,
    )
    logging.info(
        f"Producer finished, put {multiprocess_running_dict['num_func_taken_from_programdb']} functions in the queue."
    )

    # Wait for the consumers to finish and clean up
    consumers_finish_and_cleanup(
        cfg,
        consumers,
        function_classes_to_eval,
    )

    # Collect the results, will be registered at the end of the round
    passed_function_classes = copy.deepcopy(list(passed_function_classes_global))
    failed_function_classes = copy.deepcopy(list(failed_function_classes_global))

    logging.info("###################")
    logging.info(f"Num of passed functions on {dataset} set: {len(passed_function_classes)}")
    logging.info(f"Num of failed functions on {dataset} set: {len(failed_function_classes)}")
    logging.info("###################")

    # Clean up
    passed_function_classes_global[:] = []
    failed_function_classes_global[:] = []

    multiprocess_running_dict = dict(multiprocess_running_dict)
    eval_wandb_dict[f"{dataset}/num_func_taken_from_programdb"] = multiprocess_running_dict[
        "num_func_taken_from_programdb"
    ]

    manager.shutdown()
    del passed_function_classes_global
    del failed_function_classes_global
    del multiprocess_running_dict
    del manager

    if termination_event.is_set():
        raise RuntimeError("Termination event was set, there was an error.")

    eval_wandb_dict[f"{dataset}/num_func_passed_on_eval"] = len(passed_function_classes)
    eval_wandb_dict[f"{dataset}/num_func_failed_on_eval"] = len(failed_function_classes)

    # Process the passed_function_classes
    all_scores = []
    train_scores = []
    eval_scores = []
    # all_true_scores = []
    for function_class in passed_function_classes:
        train_scores.append(function_class.original_score)
        eval_scores.append(function_class.true_score)
        all_scores.append(function_class.score)
        # all_true_scores.append(function_class.true_score)

    # Get overall best 10 and 50 scores
    best_10_scores = np.sort(all_scores)[::-1][:10]
    best_10_scores_avg_overall = np.mean(best_10_scores)

    best_50_scores = np.sort(all_scores)[::-1][:50]
    best_50_scores_avg_overall = np.mean(best_50_scores)

    eval_wandb_dict[f"{dataset}/Best score overall"] = np.max(all_scores)
    eval_wandb_dict[f"{dataset}/Best 10 scores overall"] = best_10_scores_avg_overall
    eval_wandb_dict[f"{dataset}/Best 50 scores overall"] = best_50_scores_avg_overall

    eval_wandb_dict[f"{dataset}/Number of unique scores"] = len(set(all_scores))
    eval_wandb_dict[f"{dataset}/Score threshold of best 1% of programs"] = np.percentile(
        all_scores, 99
    )
    eval_wandb_dict[f"{dataset}/Score threshold of best 10% of programs"] = np.percentile(
        all_scores, 90
    )
    eval_wandb_dict[f"{dataset}/Score threshold of best 20% of programs"] = np.percentile(
        all_scores, 80
    )
    eval_wandb_dict[f"{dataset}/Score threshold of best 40% of programs"] = np.percentile(
        all_scores, 60
    )

    end_time = time.time()
    eval_wandb_dict[f"{dataset}/time_taken_to_evaluate"] = end_time - start_time
    eval_wandb_dict[f"round_num"] = round_num
    wandb.log(eval_wandb_dict)

    metrics = {}
    metrics["round_num"] = round_num
    metrics[f"dataset"] = dataset
    metrics["all_scores"] = all_scores
    metrics["train_eval_scores"] = (train_scores, eval_scores)
    metrics["best_overall_score"] = np.max(all_scores)
    metrics["best_10_scores_avg_overall"] = best_10_scores_avg_overall
    metrics["best_50_scores_avg_overall"] = best_50_scores_avg_overall
    metrics["num_unique_scores"] = len(set(all_scores))
    metrics["score_threshold_best_1_percent"] = np.percentile(all_scores, 99)
    metrics["score_threshold_best_10_percent"] = np.percentile(all_scores, 90)
    metrics["score_threshold_best_20_percent"] = np.percentile(all_scores, 80)
    metrics["score_threshold_best_40_percent"] = np.percentile(all_scores, 60)
    metrics["time_taken_to_evaluate"] = end_time - start_time
    if not os.path.exists(f"{cfg.logs_dir}/metrics"):
        os.makedirs(f"{cfg.logs_dir}/metrics")
    write_to_json(f"{cfg.logs_dir}/metrics/metrics_{dataset}_round_{round_num}.json", metrics)
    print(metrics)

    return passed_function_classes


def write_to_json(json_file, data):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Error reading JSON file {json_file}")
    else:
        all_data = []
    all_data.append(data)
    with open(json_file, "w") as f:
        json.dump(all_data, f, indent=2)
    return
