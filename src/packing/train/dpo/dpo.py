from trl import DPOConfig, DPOTrainer
import copy
from peft import PeftModel, get_peft_model
import wandb
from packing.model.learning_rate import learning_rate_schedule
from packing.utils.dataset_utils import calculate_dataset_statistics
import logging


def prepare_dpo_chats(cfg, dpo_chats, running_dict):
    assert len(dpo_chats.scores_since_finetune) > 0, "No data to finetune division by zero"
    assert len(dpo_chats.train_dataset["chosen_chat_score"]) > 0, "No data to finetune"

    running_dict["traindata/before_filtering_num_data"] = len(
        dpo_chats.train_dataset["chosen_chat_score"]
    )

    scores_since_tuning = dpo_chats.scores_since_finetune
    scores = dpo_chats.train_dataset["chosen_chat_score"]

    running_dict, score_percentile = calculate_dataset_statistics(cfg, running_dict, scores, scores_since_tuning,
                                                                  algo="dpo")

    dpo_chats.scores_since_finetune = []
    # Return the score_percentile since beginning
    return dpo_chats, running_dict, score_percentile


def DPO(
        cfg,
        running_dict,
        dpo_chats,
        score_threshold,
        model,
        tokenizer,
        model_name,
        train_run,
        lora_config,
        round_num,
):
    high_score = dpo_chats.get_highest_score()
    score_increment = (high_score - score_threshold) / cfg.max_loops

    # With model_name to catch bugs
    running_dict[f"traindata_loop/num_datapoints_{model_name}"] = []
    running_dict[f"traindata_loop/score_threshold_{model_name}"] = []
    if f"loop_num" not in running_dict:
        last_loop_num = 0
    else:
        last_loop_num = (
                copy.deepcopy(running_dict[f"{model_name}_loop_num"][-1]) + 1
        )  # because tuning_loop starts at 0
    running_dict[f"{model_name}_loop_num"] = []

    tuning_loop = 0
    while tuning_loop < cfg.max_loops:
        print(f"Finetuning score threshold: {score_threshold}")
        dpo_train_dataset = dpo_chats.get_dataset_above_threshold(score_threshold)
        num_datapoints = len(dpo_train_dataset)

        assert num_datapoints > 0, "No chats above score threshold"

        if tuning_loop == 0:
            running_dict["traindata/score_threshold"] = copy.deepcopy(score_threshold)
            running_dict["traindata/num_datapoints"] = num_datapoints
        running_dict[f"traindata_loop/num_datapoints_{model_name}"].append(num_datapoints)
        running_dict[f"traindata_loop/score_threshold_{model_name}"].append(copy.deepcopy(score_threshold))
        running_dict[f"{model_name}_loop_num"].append(last_loop_num + tuning_loop)

        wandb.log(
            {"traindata/num_datapoints": num_datapoints, "traindata/score_threshold": copy.deepcopy(score_threshold)})

        learning_rate = learning_rate_schedule(cfg, cfg.train.dpo_config.learning_rate, num_datapoints, tuning_loop,
                                               round_num)

        training_args = DPOConfig(
            beta=cfg.train.dpo_config.beta,
            max_length=cfg.train.dpo_config.max_seq_length,
            max_prompt_length=cfg.train.dpo_config.max_seq_length - cfg.model.max_tokens,
            per_device_train_batch_size=cfg.train.dpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.train.dpo_config.gradient_accumulation_steps,
            gradient_checkpointing=bool(cfg.train.dpo_config.gradient_checkpointing),
            gradient_checkpointing_kwargs={
                "use_reentrant": False} if cfg.train.dpo_config.gradient_checkpointing else None,
            num_train_epochs=cfg.train.dpo_config.num_train_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=cfg.train.dpo_config.lr_scheduler_type,
            weight_decay=cfg.train.dpo_config.weight_decay,
            warmup_steps=cfg.train.dpo_config.warmup_steps,
            logging_steps=cfg.train.dpo_config.logging_steps,
            bf16=True,
            logging_first_step=True,
            remove_unused_columns=True,
            save_strategy="no",
            output_dir=cfg.logs_dir,
            report_to="wandb" if cfg.wandb else "none",
            seed=cfg.seed,
            max_grad_norm=1.0,
            f_divergence_type=cfg.train.dpo_config.f_divergence_type,
            f_alpha_divergence_coef=cfg.train.dpo_config.f_alpha_divergence_coef,
        )

        logging.info(
            f"Training with divergence {cfg.train.dpo_config.f_divergence_type} and alpha {cfg.train.dpo_config.f_alpha_divergence_coef}")

        # Shuffle the dataset
        dpo_train_dataset = dpo_train_dataset.shuffle(seed=cfg.seed)

        print(f"Finetuning with {len(dpo_train_dataset)} chats")
        if isinstance(model, PeftModel):
            model = model.merge_and_unload()
            del model.peft_config
        model.config.use_cache = False
        model.train()
        # model.print_trainable_parameters()
        trainer = DPOTrainer(
            model,
            ref_model=None,
            args=training_args,
            train_dataset=dpo_train_dataset,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        trainer.train()

        if cfg.one_tuning:
            break

        model = trainer.model
        tuning_loop += 1
        score_threshold += score_increment

    return trainer, running_dict, train_run
