import os

import torch
import wandb
from omegaconf import OmegaConf
import logging

from packing.model.model import initialize_single_model, clean_up_gpu_mem
from packing.model.sft import SFT
from packing.model.dpo import DPO
from packing.model.both import SFT_DPO
from peft import LoraConfig
import copy
import pickle
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from packing.utils.seeding import seed_everything
from transformers import set_seed

def train_model(
    cfg,
    running_dict,
    model_name,
    full_model_name,
    model_adapter_dir,
    round_num,
    sft_chats,
    sft_threshold,
    dpo_chats,
    dpo_threshold,
):
    set_seed(cfg.seed)

    assert (
        cfg.sft or cfg.dpo or cfg.both
    ), "At least one of the flags sft, dpo, or both must be True"

    if cfg.wandb:
        train_run = wandb.init(
            project=f"{cfg.project}-{cfg.prefix}-train",
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=f"{cfg.wandb_name}_round{round_num}_{model_name}",
            group=f"{cfg.group_name}_round{round_num}_{model_name}",
            reinit=False,
            entity=cfg.entity,
        )
    logging.info(f"Training model {model_name}")

    logging.info(
        f"  --> Memory before downloading finetuning model: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
    )

    model, tokenizer, _ = initialize_single_model(
        cfg, full_model_name, load_finetuned=False, train=True
    )

    logging.info(
        f"  --> Memory after downloading finetuning model: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
    )

    target_modules = "all-linear"
    # target_modules = [ "q_proj",
    #     "k_proj",
    #     "v_proj",
    # ]
    lora_config = LoraConfig(
        r=cfg.lora_config.r,
        lora_alpha=cfg.lora_config.lora_alpha,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    if cfg.sft:
        logging.info("-" * 10)
        logging.info("SFT FINETUNING")

        trainer, running_dict, train_run = SFT(
            cfg,
            running_dict,
            sft_chats,
            sft_threshold,
            model,
            tokenizer,
            model_name,
            train_run,
            lora_config,
            round_num,
        )

    elif cfg.dpo:
        logging.info("-" * 10)
        logging.info("DPO FINETUNING")
        trainer, running_dict, train_run = DPO(
            cfg,
            running_dict,
            dpo_chats,
            dpo_threshold,
            model,
            tokenizer,
            model_name,
            train_run,
            lora_config,
            round_num,
        )
    elif cfg.both:
        logging.info("-" * 10)
        logging.info("FINETUNING WITH SFT AND DPO")

        trainer, running_dict, train_run = SFT_DPO(
            cfg,
            running_dict,
            sft_chats,
            sft_threshold,
            dpo_chats,
            dpo_threshold,
            model,
            tokenizer,
            model_name,
            train_run,
            lora_config,
            round_num,
        )

    logging.info(f"Training finished, saving model {model_name} in {model_adapter_dir}")
    # trainer.save_model(model_adapter_dir)

    # Saving the full model since multiple adapters are used sequentially on top of the base model
    finetuned_model = trainer.model
    finetuned_model_merged = finetuned_model.merge_and_unload(progressbar=True, safe_merge=True)
    del finetuned_model_merged.peft_config
    finetuned_model_merged.save_pretrained(model_adapter_dir)
    # Save the tokenizer as well, save newly initialized tokenizer, to not have to deal with pad and eos tokens
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    tokenizer.save_pretrained(model_adapter_dir)

    del trainer.model
    del trainer.optimizer
    del trainer.lr_scheduler
    del trainer.accelerator
    del trainer

    logging.info(
        f"  --> Memory after training {model_name}: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
    )

    if cfg.wandb:
        train_run.finish()

    model = model.cpu()
    del model
    del tokenizer
    clean_up_gpu_mem()
    logging.info(
        f"  --> Memory after deleting the finetuned model: {torch.cuda.max_memory_allocated() // 1024 // 1024}MB"
    )
    return running_dict


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the model with specified arguments.")
    parser.add_argument("--logs_dir", type=str, help="Directory to read cfg from")
    parser.add_argument("--model_name", type=str, help="Model name")
    parser.add_argument("--full_model_name", type=str, help="Full model name")
    parser.add_argument("--model_adapter_dir", type=str, help="Model adapter directory")
    parser.add_argument("--round_num", type=int, help="Round number")
    parser.add_argument("--sft_threshold", type=float, help="SFT threshold")
    parser.add_argument("--dpo_threshold", type=float, help="DPO threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Read cfg, sft_chats, dpo_chats from the args.cfg_logs_dir
    cfg = OmegaConf.load(f"{args.logs_dir}/config.yaml")

    with open(f"{cfg.logs_dir}/sft_chats_train.pkl", "rb") as f:
        sft_chats = pickle.load(f)
        assert sft_chats is not None

    with open(f"{cfg.logs_dir}/dpo_chats_train.pkl", "rb") as f:
        dpo_chats = pickle.load(f)
        assert dpo_chats is not None

    with open(f"{cfg.logs_dir}/running_dict.pkl", "rb") as f:
        running_dict = pickle.load(f)
        assert running_dict is not None

    assert running_dict is not None
    running_dict = train_model(
        cfg,
        running_dict,
        args.model_name,
        args.full_model_name,
        args.model_adapter_dir,
        args.round_num,
        sft_chats,
        args.sft_threshold,
        dpo_chats,
        args.dpo_threshold,
    )

    assert running_dict is not None
    with open(f"{cfg.logs_dir}/running_dict.pkl", "wb") as f:
        assert running_dict is not None
        pickle.dump(running_dict, f)


#### SCRIPT BELOW FOR TESTING ONLY THE TRAINING FUNCTION ####
# if __name__ == "__main__":
#     """
#     Test the training script
#     """
#     import pickle
#     from packing.model.model import initialize_single_model, get_full_model_name, initialize_models
#     from packing.logging.logging import load_config

#     # Set up logging
#     logger = logging.getLogger()
#     logger.setLevel(logging.INFO)
#     #file_handler = logging.FileHandler(f"{cfg.logs_dir}/stdout.log")
#     #file_handler.setLevel(logging.INFO)
#     console_handler = logging.StreamHandler()
#     console_handler.setLevel(logging.INFO)
#     formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
#     #file_handler.setFormatter(formatter)
#     console_handler.setFormatter(formatter)
#     #logger.addHandler(file_handler)
#     logger.addHandler(console_handler)

#     default = OmegaConf.create(
#         {
#             "config_common": "taskbin",
#             "config_specific": "specific",
#             "seed": 0,
#             "prefix": "debug",
#             "config_folder": "configs",
#         }
#     )
#     config_cli = OmegaConf.from_cli()
#     config_cli = OmegaConf.merge(
#         default, config_cli
#     )
#     cfg = load_config(config_cli)
#     # Perform checks on the config to ensure it is valid
#     assert cfg.task_bin or cfg.task_code or cfg.task_tsp, "One task must be selected"
#     assert not (cfg.task_bin and cfg.task_code and cfg.task_tsp), "Only one task can be selected"
#     if not cfg.one_tuning:
#         assert cfg.max_loops > 1, "If not one_tuning, max_loops must be greater than 1"
#     if cfg.task_bin:
#         cfg.task_name = "bin"
#     elif cfg.task_code:
#         cfg.task_name = "code"
#     elif cfg.task_tsp:
#         cfg.task_name = "tsp"
#     assert (cfg.sft + cfg.dpo + cfg.both) < 2, "Only one training method can be selected"
#     if cfg.task_bin:
#         assert not (cfg.Weibull and cfg.OR), "Only one bin packing dataset can be selected"

#     cfg.logs_dir = (
#         f"logs/{cfg.prefix}/{cfg.task_name}_{cfg.config_common}_{cfg.config_specific}_{cfg.seed}"
#     )
#     cfg.wandb_name = f"{cfg.prefix}/{cfg.config_common}_{cfg.config_specific}_{cfg.seed}"
#     cfg.group_name = f"{cfg.prefix}/{cfg.config_common}_{cfg.config_specific}"

#     # Model specific cfg
#     cfg.full_model_name = get_full_model_name(cfg)
#     if torch.cuda.get_device_properties(0).major < 8:
#         cfg.model_dtype = "float16"
#     else:
#         cfg.model_dtype = "bfloat16"

#     if cfg.multiple_models:
#         cfg.model_adapter_dir = [
#             f"{cfg.logs_dir}/model_adapter_{model_name}" for model_name in cfg.model_name
#         ]
#     else:
#         cfg.model_adapter_dir = f"{cfg.logs_dir}/model_adapter_{cfg.model_name}"

#     cfg.full_model_name = get_full_model_name(cfg)

#     #model, tokenizer, sampling_params = initialize_models(cfg, load_finetuned=False)

#     with open(f"{cfg.logs_dir}/sft_chats.pkl", "rb") as f:
#         sft_chats = pickle.load(f)

#     with open(f"{cfg.logs_dir}/dpo_chats.pkl", "rb") as f:
#         dpo_chats = pickle.load(f)

#     # Read the dpo_chats and sft_chats from cfg.output_dir
#     running_dict = {}

#     round_num=0,
#     sft_chats=sft_chats
#     sft_threshold=-1000
#     dpo_chats=dpo_chats
#     dpo_threshold=-1000

#     deduped = []
#     seen_models = set()
#     for mn, fmn, mad, gpu in zip(cfg.model_name, cfg.full_model_name, cfg.model_adapter_dir, cfg.gpu_nums):
#         if mn not in seen_models:
#             seen_models.add(mn)
#             deduped.append((mn, fmn, mad, gpu))

#     for model_name, full_model_name, model_adapter_dir, gpu_num in deduped:
#         # with set_cuda_visible_devices('0'):
#         train_model(
#             cfg,
#             running_dict,
#             model_name,
#             full_model_name,
#             model_adapter_dir,
#             round_num,
#             copy.deepcopy(sft_chats),
#             copy.deepcopy(sft_threshold),
#             copy.deepcopy(dpo_chats),
#             copy.deepcopy(dpo_threshold),
#         )
#     return running_dict
