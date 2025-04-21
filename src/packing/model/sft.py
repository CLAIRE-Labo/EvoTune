from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import copy
import wandb
import pandas as pd
from peft import get_peft_model
import torch
from packing.model.learning_rate import learning_rate_schedule
import numpy as np
from packing.model.dataset_utils import calculate_dataset_statistics


def prepare_sft_chats(cfg, sft_chats, running_dict):
    assert len(sft_chats.data) > 0, "No data to finetune"
    assert len(sft_chats.scores_since_finetune) > 0, "No data to finetune division by zero"

    running_dict["traindata/before_filtering_num_data"] = len(sft_chats.data)
    print(f"Finetuning data before filtering has total {len(sft_chats.data)} chats")

    scores_since_tuning = sft_chats.scores_since_finetune
    scores = sft_chats.scores

    running_dict, score_percentile = calculate_dataset_statistics(cfg, running_dict, scores, scores_since_tuning, algo="sft")

    sft_chats.scores_since_finetune = []
    # Return the score_percentile since beginning

    return sft_chats, running_dict, score_percentile

    # #sorted_scores = sorted(sft_chats.scores, reverse=True)
    # # percentile = cfg.percentile * 0.01
    # # score_percentile = sorted_scores[int(len(sorted_scores) * percentile)]
    # score_percentile = np.percentile(scores, (100-cfg.percentile))
    # running_dict["traindata/sft_score_percentile"] = score_percentile

    # median = np.percentile(scores, 50)
    # running_dict["traindata/sft_score_median"] = median

    # sft_chats.scores_since_finetune = []


def SFT(
    cfg,
    running_dict,
    sft_chats,
    score_threshold,
    model,
    tokenizer,
    model_name,
    train_run,
    lora_config,
    round_num,
):
    model = get_peft_model(model, lora_config)

    high_score = sft_chats.get_highest_score()
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
        sft_train_dataset = sft_chats.get_dataset_above_threshold(score_threshold)
        num_datapoints = len(sft_train_dataset)
        assert num_datapoints > 0, "No chats above score threshold"

        if tuning_loop == 0:
            running_dict["traindata/score_threshold"] = copy.deepcopy(score_threshold)
            running_dict["traindata/num_datapoints"] = num_datapoints

        running_dict[f"traindata_loop/num_datapoints_{model_name}"].append(num_datapoints)
        running_dict[f"traindata_loop/score_threshold_{model_name}"].append(score_threshold)
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

        print(f"Finetuning with {len(sft_train_dataset)} chats")
        model.print_trainable_parameters()
        model.train()
        trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=sft_train_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        if cfg.one_tuning:
            break

        model = trainer.model
        # TODO:Merge the adapter and add a new adapter
        # model.merge_and_unload_model()
        tuning_loop += 1
        score_threshold += score_increment

    return trainer, running_dict, train_run

    # # Log every loop separately
    # if cfg.wandb:
    #     train_run.finish()
    #     train_run = wandb.init(
    #         project=f"{cfg.project}-{cfg.prefix}-train",
    #         config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
    #         name=f"{cfg.wandb_name}_round{round_num}_{model_name}_loop{tuning_loop}",
    #         group=f"{cfg.group_name}_round{round_num}_{model_name}_loop{tuning_loop}",
    #         reinit=False,
    #         entity=cfg.entity,
    #     )

    # def custom_preprocess(ds):
    #     def proces_sft(row):
    #         row = train_dataset[0]
    #         # row = {"messages":[{"role":"system", "content": "This is a system prompt."},
    #         # {"role": "user", "content": "Hello, how are you? "},
    #         # {"role": "assistant", "content": " I'm doing great. How can I help you today?"}
    #         # ]}
    #         prompt_tokens = tokenizer.apply_chat_template(row["messages"][:-1], add_generation_prompt=True, tokenize=True)
    #         chat_tokens = tokenizer.apply_chat_template(row["messages"], tokenize=True)
    #         # Assert last prompt token hasn't been merged
    #         assert chat_tokens[:len(prompt_tokens)] == prompt_tokens
    #         prompt = tokenizer.decode(prompt_tokens)
    #         chat = tokenizer.decode(chat_tokens[:cfg.sft_config.max_seq_length])
    #         # Assert prompt hasn't been truncated or modified (last token merged)
    #         assert prompt in chat
    #         res = {
    #             "prompt_len": len(prompt_tokens),
    #             "chat_original_len": len(chat_tokens),
    #             "max_len": len(chat_tokens),
    #             "chat": chat,
    #             "prompt": prompt,
    #             "completion": chat[len(prompt):],
    #             "input_ids": chat_tokens[:cfg.sft_config.max_seq_length],
    #             "attention_mask": [1] * len(chat_tokens[:cfg.sft_config.max_seq_length]),
    #         }
    #         return res
