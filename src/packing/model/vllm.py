

# def delete_vllm():
#     del model.llm_engine.model_executor.driver_worker
#     del model.llm_engine.model_executor
#     del model.llm_engine
#     del model

# def initialize_model_vllm(cfg, full_model_name, training_flag, flag_load_finetuned):
#     if not training_flag:
#         # Initialize the VLLM model for fast inference
#         if flag_load_finetuned or cfg.compare_train:
#             # Load the model with LORA enabled
#             model = LLM(
#                 model=full_model_name,
#                 tokenizer=full_model_name,
#                 enable_lora=True,
#                 max_model_len=10000,
#                 dtype=cfg.model_dtype,
#                 seed=cfg.seed,
#                 max_loras=1,
#                 max_lora_rank=cfg.lora_config.r,
#                 max_seq_len_to_capture=48000,
#             )  # , max_num_batched_tokens=65528, max_model_len=65528)

#         else:
#             # No finetuning happened yet, basic VLLM model
#             if cfg.enforce_eager:
#                 model = LLM(
#                     model=full_model_name,
#                     tokenizer=full_model_name,
#                     max_model_len=10000,
#                     tensor_parallel_size=1,
#                     dtype=cfg.model_dtype,
#                     seed=cfg.seed,
#                     enforce_eager=True,
#                 )
#             else:
#                 model = LLM(
#                     model=full_model_name,
#                     tokenizer=full_model_name,
#                     max_model_len=10000,
#                     tensor_parallel_size=1,
#                     dtype=cfg.model_dtype,
#                     seed=cfg.seed,
#                     enforce_eager=False,
#                 )

#         tokenizer = model.llm_engine.tokenizer.tokenizer
#         # tokenizer.padding_side='right'
#         sampling_params = SamplingParams(
#             n=cfg.num_outputs_per_prompt,
#             max_tokens=cfg.max_tokens,
#             temperature=cfg.temperature,
#             top_k=cfg.topk,
#             top_p=cfg.topp,
#             seed=cfg.seed,
#         )
#     return model, tokenizer, sampling_params


# def query_model_vllm(cfg, chat, model, tokenizer, sampling_params, flag_load_finetuned):
#     time_start = time.time()
#     if cfg.vllm:
#         input_ids = tokenizer.apply_chat_template(chat, add_generation_prompt=True)
#         len_input_token_ids = len(input_ids)
#         # input_ids_debug = tokenizer_debug.apply_chat_template(chat, add_generation_prompt=True)
#         # assert input_ids == input_ids_debug

#         logging.info("Tokenization finished, generating...")
#         # Print percentage of cuda memory used
#         logging.info(f"cuda memory allocated: {torch.cuda.memory_allocated() // 1024 // 1024}MB")
#         if flag_load_finetuned:
#             lora_request = LoRARequest("adapter", 1, lora_path=cfg.model_adapter_dir)
#             output_dict = model.generate(
#                 prompt_token_ids=[input_ids],
#                 sampling_params=sampling_params,
#                 lora_request=lora_request,
#             )
#             # lora_request_debug = LoRARequest("sql_adapter", 1, lora_path ='logs/debug/bin_common_specific_0/checkpoint-1')
#             # output_dict_debug = model.generate(prompt_token_ids=[input_ids], sampling_params=sampling_params, lora_request=lora_request_debug)
#         else:
#             output_dict = model.generate(
#                 prompt_token_ids=[input_ids], sampling_params=sampling_params
#             )
#         print("Model finished generating")
#         outputs_text_init = [
#             output_dict[0].outputs[i].text for i in range(len(output_dict[0].outputs))
#         ]
#         print("Checkpoint 0")
#         len_output_ids_init = [
#             len(output_dict[0].outputs[i].token_ids) for i in range(len(output_dict[0].outputs))
#         ]
#         print("Checkpoint 1")
#         # output_ids_init = [output_dict[0].outputs[i].token_ids for i in range(len(output_dict[0].outputs))]

#     # The rest of the code is the same as for the Huggingface model
#     raise NotImplementedError("VLLM model not implemented")
