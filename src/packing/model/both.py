from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer
from trl import DPOConfig, DPOTrainer
from datasets import Dataset
import copy
import wandb
from peft import PeftModel, get_peft_model
from packing.model.learning_rate import learning_rate_schedule
import logging


def SFT_DPO(
    cfg,
    running_dict,
    sft_chats,
    sft_score_threshold,
    dpo_chats,
    dpo_score_threshold,
    model,
    tokenizer,
    model_name,
    train_run,
    lora_config,
    round_num,
):

    high_score_sft = sft_chats.get_highest_score()
    score_increment_sft = (high_score_sft - sft_score_threshold) / cfg.max_loops
    high_score_dpo = dpo_chats.get_highest_score()
    score_increment_dpo = (high_score_dpo - dpo_score_threshold) / cfg.max_loops

    # With model_name to catch bugs
    running_dict[f"traindata_loop/sft_num_datapoints_{model_name}"] = []
    running_dict[f"traindata_loop/dpo_num_datapoints_{model_name}"] = []
    running_dict[f"traindata_loop/sft_score_threshold_{model_name}"] = []
    running_dict[f"traindata_loop/dpo_score_threshold_{model_name}"] = []
    if f"loop_num" not in running_dict:
        last_loop_num = 0
    else:
        last_loop_num = (
            copy.deepcopy(running_dict[f"{model_name}_loop_num"][-1]) + 1
        )  # because tuning_loop starts at 0
    running_dict[f"{model_name}_loop_num"] = []

    tuning_loop = 0
    while tuning_loop < cfg.max_loops:
        # ======================= SFT =======================
        print(f"SFT Finetuning score threshold: {sft_score_threshold}")
        sft_train_dataset = sft_chats.get_dataset_above_threshold(sft_score_threshold)
        num_datapoints = len(sft_train_dataset)
        assert num_datapoints > 0, "No chats above score threshold for SFT"

        if tuning_loop == 0:
            running_dict["traindata/sft_score_threshold"] = copy.deepcopy(sft_score_threshold)
            running_dict["traindata/sft_num_datapoints"] = num_datapoints

        running_dict[f"traindata_loop/sft_num_datapoints_{model_name}"].append(num_datapoints)
        running_dict[f"traindata_loop/sft_score_threshold_{model_name}"].append(sft_score_threshold)
        running_dict[f"{model_name}_loop_num"].append(
            last_loop_num + tuning_loop
        )  # * cfg.sft_config.num_train_epochs

        learning_rate = learning_rate_schedule(cfg, cfg.sft_config.learning_rate, num_datapoints, tuning_loop, round_num)

        # torch.cuda.set_per_process_memory_fraction(0.98, torch.cuda.current_device())
        sft_config = SFTConfig(
            max_seq_length=cfg.sft_config.max_seq_length,
            per_device_train_batch_size=cfg.sft_config.per_device_train_batch_size,  # Adjust this according to your GPU memory
            gradient_accumulation_steps=cfg.sft_config.gradient_accumulation_steps,
            gradient_checkpointing=bool(cfg.sft_config.gradient_checkpointing),
            num_train_epochs=cfg.sft_config.num_train_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=cfg.sft_config.lr_scheduler_type,
            weight_decay=cfg.sft_config.weight_decay,
            warmup_steps=cfg.sft_config.warmup_steps,
            logging_steps=cfg.sft_config.logging_steps,
            bf16=True,
            logging_first_step=True,
            save_strategy="no",
            output_dir=cfg.logs_dir,
            report_to="wandb" if cfg.wandb else "none",
            seed=cfg.seed,
            max_grad_norm=1.0,
            # optim=cfg.sft_config.optim,
            # gradient_checkpointing_kwargs={
            #     "use_reentrant": False if cfg.gradient_checkpointing else True
            # },
        )

        # Shuffle the dataset
        sft_train_dataset = sft_train_dataset.shuffle(seed=cfg.seed)

        # Remove duplicates from the dataset
        df = sft_train_dataset.to_pandas()
        df = df.drop_duplicates()
        sft_train_dataset = Dataset.from_pandas(df)

        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            del model.peft_config
        print(f"SFT Finetuning with {len(sft_train_dataset)} chats, round {tuning_loop}, isinstance(model, PeftModel): {isinstance(model, PeftModel)}")
        # model.print_trainable_parameters() # print trainable parameters no longer callable as model is not a PeftModel
        model.train()
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=sft_train_dataset,
            tokenizer=tokenizer,
            peft_config=lora_config,
        )
        trainer.train()

        # should be done only after one round of SFT and DPO
        # if cfg.one_tuning:
        #     break

        # sft_peft_model = copy.deepcopy(trainer.model)
        sft_peft_model = trainer.model
        sft_score_threshold += score_increment_sft

        # ======================= DPO =======================
        print(f"DPO Finetuning score threshold: {dpo_score_threshold}")
        dpo_train_dataset = dpo_chats.get_dataset_above_threshold(dpo_score_threshold)
        num_datapoints = len(dpo_train_dataset)
        assert num_datapoints > 0, "No chats above score threshold"

        if tuning_loop == 0:
            running_dict["traindata/dpo_score_threshold"] = copy.deepcopy(dpo_score_threshold)
            running_dict["traindata/dpo_num_datapoints"] = num_datapoints
        running_dict[f"traindata_loop/dpo_num_datapoints_{model_name}"].append(num_datapoints)
        running_dict[f"traindata_loop/dpo_score_threshold_{model_name}"].append(dpo_score_threshold)
        # running_dict[f"{model_name}_loop_num"].append(last_loop_num + tuning_loop)

        learning_rate = learning_rate_schedule(cfg, cfg.dpo_config.learning_rate, num_datapoints, tuning_loop, round_num)

        # torch.cuda.set_per_process_memory_fraction(0.98, torch.cuda.current_device())
        training_args = DPOConfig(
            beta=cfg.dpo_config.beta,
            max_length=cfg.dpo_config.max_seq_length,
            max_prompt_length=cfg.dpo_config.max_seq_length - cfg.max_tokens,
            per_device_train_batch_size=cfg.dpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.dpo_config.gradient_accumulation_steps,
            gradient_checkpointing=bool(cfg.dpo_config.gradient_checkpointing),
            num_train_epochs=cfg.dpo_config.num_train_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=cfg.dpo_config.lr_scheduler_type,
            weight_decay=cfg.dpo_config.weight_decay,
            warmup_steps=cfg.dpo_config.warmup_steps,
            logging_steps=cfg.dpo_config.logging_steps,
            bf16=True,
            logging_first_step=True,
            remove_unused_columns=True,
            save_strategy="no",
            output_dir=cfg.logs_dir,
            report_to="wandb" if cfg.wandb else "none",
            seed=cfg.seed,
            max_grad_norm=1.0,
            f_divergence_type=cfg.dpo_config.f_divergence_type,
            f_alpha_divergence_coef=cfg.dpo_config.f_alpha_divergence_coef,
        )
        # Shuffle the dataset
        dpo_train_dataset = dpo_train_dataset.shuffle(seed=cfg.seed)

        logging.info("Merge Lora")
        merged_model = sft_peft_model.merge_and_unload()
        del merged_model.peft_config
        logging.info(f"DPO Finetuning with {len(dpo_train_dataset)} chats, round {tuning_loop}, isinstance(model, PeftModel): {isinstance(merged_model, PeftModel)}")
        merged_model.train()
        trainer = DPOTrainer(model=merged_model
                        , ref_model=None
                        , args=training_args
                        , train_dataset=dpo_train_dataset
                        , tokenizer=tokenizer
                        , peft_config=lora_config
                        )
        trainer.train()
        # model = copy.deepcopy(trainer.model)
        model = trainer.model

        if cfg.one_tuning:
            break

        tuning_loop += 1
        dpo_score_threshold += score_increment_dpo

        del sft_peft_model

    del merged_model
    del model
    
    return trainer, running_dict, train_run
