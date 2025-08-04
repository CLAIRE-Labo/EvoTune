import torch
import wandb
from omegaconf import OmegaConf
import logging

from packing.model.model import initialize_single_model, clean_up_gpu_mem
from packing.train.dpo.dpo import DPO
from peft import LoraConfig
import pickle
import argparse
from transformers import AutoTokenizer
from transformers import set_seed


def train_model(
        cfg,
        running_dict,
        model_name,
        full_model_name,
        model_adapter_dir,
        round_num,
        dpo_chats,
        dpo_threshold,
):
    set_seed(cfg.seed)

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
    if cfg.train.train_method_name.lower() == 'dpo':
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
    else:
        raise Exception("This training method has not yet been implemented")

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
    parser.add_argument("--dpo_threshold", type=float, help="DPO threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Read cfg, dpo_chats from the args.cfg_logs_dir
    cfg = OmegaConf.load(f"{args.logs_dir}/config.yaml")

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
        dpo_chats,
        args.dpo_threshold,
    )

    assert running_dict is not None
    with open(f"{cfg.logs_dir}/running_dict.pkl", "wb") as f:
        assert running_dict is not None
        pickle.dump(running_dict, f)
