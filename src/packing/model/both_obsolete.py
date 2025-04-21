from trl import SFTTrainer
from trl import SFTConfig, SFTTrainer
from datasets import Dataset
import copy
import wandb
import pandas as pd
from trl import DPOConfig, DPOTrainer
from peft import PeftModel


### NOT FUNCTIONAL YET

def BOTH(
    cfg,
    running_dict,
    sft_chats,
    dpo_chats,
    score_threshold,
    model,
    tokenizer,
    model_name,
    train_run,
):

    high_score = sft_chats.get_highest_score()
    score_increment = (high_score - score_threshold) / cfg.max_loops

    # With model_name to catch bugs
    running_dict[f"traindata_loop/score_threshold_{model_name}"] = []
    running_dict[f"traindata_loop/sft_num_datapoints_{model_name}"] = []
    running_dict[f"traindata_loop/dpo_num_datapoints_{model_name}"] = []
    
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
        dpo_train_dataset = dpo_chats.get_dataset_above_threshold(score_threshold)

        assert len(sft_train_dataset) > 0, "No sft chats above score threshold"
        assert len(dpo_train_dataset) > 0, "No dpo chats above score threshold"

        if tuning_loop == 0:
            running_dict["traindata/score_threshold"] = copy.deepcopy(score_threshold)
            running_dict["traindata/sft_num_datapoints"] = len(sft_train_dataset)
            running_dict["traindata/dpo_num_datapoints"] = len(dpo_train_dataset)
        running_dict[f"traindata_loop/score_threshold_{model_name}"].append(score_threshold)
        running_dict[f"traindata_loop/sft_num_datapoints_{model_name}"].append(len(sft_train_dataset))
        running_dict[f"traindata_loop/dpo_num_datapoints_{model_name}"].append(len(dpo_train_dataset))
        running_dict[f"{model_name}_loop_num"].append(
            last_loop_num + tuning_loop
        )  # * cfg.sft_config.num_train_epochs

        if cfg.lr_annealing:
            sft_learning_rate = cfg.sft_config.learning_rate / (2**tuning_loop)
            dpo_learning_rate = cfg.dpo_config.learning_rate / (2**tuning_loop)
        else:
            sft_learning_rate = cfg.sft_config.learning_rate
            dpo_learning_rate = cfg.dpo_config.learning_rate

        sft_config = SFTConfig(
            max_seq_length=cfg.sft_config.max_seq_length,
            per_device_train_batch_size=cfg.sft_config.per_device_train_batch_size,  # Adjust this according to your GPU memory
            gradient_accumulation_steps=cfg.sft_config.gradient_accumulation_steps,
            gradient_checkpointing=bool(cfg.sft_config.gradient_checkpointing),
            num_train_epochs=cfg.sft_config.num_train_epochs,
            learning_rate=sft_learning_rate,
            lr_scheduler_type=cfg.sft_config.lr_scheduler_type,
            weight_decay=cfg.sft_config.weight_decay,
            warmup_steps=cfg.sft_config.warmup_steps,
            logging_steps=cfg.sft_config.logging_steps,
            fp16=False,
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
        dpo_train_dataset = dpo_train_dataset.shuffle(seed=cfg.seed)

        # Remove duplicates from the sft dataset
        df = sft_train_dataset.to_pandas()
        df = df.drop_duplicates()
        sft_train_dataset = Dataset.from_pandas(df)

        print(f"Finetuning with {len(sft_train_dataset)} chats")
        sft_trainer = SFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=sft_train_dataset,
            tokenizer=tokenizer,
        )
        sft_trainer.train()

        # Save the SFT-trained model
        #sft_trainer.save_model("./sft_trained_model")

        # Load the trained model
        #model = PeftModel.from_pretrained("./sft_trained_model")
        ## TODO!!!!
        assert False, "Not working yet"

        dpo_config = DPOConfig(
            # max_steps=10 For testing
            beta=cfg.dpo_config.beta,
            max_length=cfg.dpo_config.max_seq_length,
            per_device_train_batch_size=cfg.dpo_config.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.dpo_config.gradient_accumulation_steps,
            gradient_checkpointing=bool(cfg.dpo_config.gradient_checkpointing),
            num_train_epochs=cfg.dpo_config.num_train_epochs,
            learning_rate=dpo_learning_rate,
            lr_scheduler_type=cfg.dpo_config.lr_scheduler_type,
            weight_decay=cfg.dpo_config.weight_decay,
            warmup_steps=cfg.dpo_config.warmup_steps,
            logging_steps=cfg.dpo_config.logging_steps,
            fp16=False,
            logging_first_step=True,
            remove_unused_columns=True,
            save_strategy="no",
            output_dir=cfg.logs_dir,
            report_to="wandb" if cfg.wandb else "none",
            seed=cfg.seed,
            max_grad_norm=1.0,
        )

        print(f"DPO Finetuning with {len(dpo_train_dataset)} chats")
        dpo_trainer = DPOTrainer(
            model= sft_trainer.model,
            ref_model=None,
            args=dpo_config,
            train_dataset=dpo_train_dataset,
            tokenizer=tokenizer,
            # peft_config=lora_config,
        )
        dpo_trainer.train()

        if cfg.one_tuning:
            break

        tuning_loop += 1
        score_threshold += score_increment

    return dpo_trainer, running_dict, train_run

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


