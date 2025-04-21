## This is an example of how two models can be placed on different GPUs using Ray and the vllm library.

def main2():
        import os
        import sys
        import socket
        import ray
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        from vllm import LLM

        # Initialize Ray
        ray.init()

        # Define the number of models (and GPUs) you want to use
        num_models = 2  # Adjust this based on your available GPUs

        # Create a placement group with one GPU and one CPU per bundle
        pg = placement_group(
            name="llm_pg",
            bundles=[{"GPU": 1, "CPU": 1} for _ in range(num_models)],
            strategy="STRICT_PACK"  # or "PACK" or "SPREAD" depending on your needs
        )
        # Wait until the placement group is ready
        ray.get(pg.ready())

        # Define the LLMActor class that will load the LLM model on the assigned GPU
        @ray.remote(num_gpus=1, num_cpus=1)
        class LLMActor:
            def __init__(self, model_name):
                import os
                import torch

                # Get the GPU IDs assigned to this actor by Ray
                gpu_ids = ray.get_gpu_ids()
                # Set CUDA_VISIBLE_DEVICES to limit the GPUs visible to this process
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(int(gpu_id)) for gpu_id in gpu_ids)
                # Set the default CUDA device
                torch.cuda.set_device(0)  # Since only one GPU is visible, it's cuda:0
                # Initialize the LLM model
                self.llm = LLM(model=model_name, device="cuda:0")  # Use cuda:0 since only one GPU is visible

            def generate(self, prompt):
                # Generate text using the LLM instance
                outputs = self.llm.generate([prompt])
                return outputs[0].outputs[0].text

        # Main function
        model_name = "gpt2"  # Replace with your model
        prompts = ["Hello from model 1", "Greetings from model 2"]

        # Create LLMActor instances assigned to different bundles in the placement group
        actors = []
        for i in range(num_models):
            # Assign the actor to a specific bundle in the placement group
            actor = LLMActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i
                )
            ).remote(model_name)
            actors.append(actor)

        # Generate outputs using the actors
        futures = []
        for actor, prompt in zip(actors, prompts):
            future = actor.generate.remote(prompt)
            futures.append(future)

        # Retrieve and print the outputs
        outputs = ray.get(futures)
        for i, output in enumerate(outputs):
            print(f"Output from model {i+1}: {output}")

main2()



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["VLLM_TARGET_DEVICE"] = "cuda:1"
# #model1 = LLM(model=cfg.full_model_name, tokenizer=cfg.full_model_name, seed=cfg.seed,tensor_parallel_size=1, device=f"cuda:0") #, max_num_batched_tokens=65528, max_model_len=65528)
# torch.cuda.synchronize() 
# from vllm import LLM, SamplingParams
# model2 = LLM(model=cfg.full_model_name, tokenizer=cfg.full_model_name, seed=cfg.seed,tensor_parallel_size=1, device=f"cuda") #, max_num_batched_tokens=65528, max_model_len=65528)
# print("placed model 2")      
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#     print(f"Memory Usage:")
#     print(f"Allocated: {torch.cuda.memory_allocated(i)/1024**3} GB")
#     print(f"Cached:    {torch.cuda.memory_reserved(i)/1024**3} GB") 


# import os
# #initialize_model_v2(cfg, training_flag=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import torch
# from vllm import LLM, SamplingParams
# #print("The cuda visible devices are: ", os.environ["CUDA_VISIBLE_DEVICES"]
# with torch.cuda.device(0):
#     model1 = LLM(model=cfg.full_model_name, tokenizer=cfg.full_model_name, seed=cfg.seed, tensor_parallel_size=0.5, device=f"cuda:0") #, max_num_batched_tokens=65528, max_model_len=65528)
# print("placed model 1")
# # print how much memory is used
# print(f"cuda memory allocated: {torch.cuda.memory_allocated() // 1024 // 1024}MB")
# torch.cuda.empty_cache()
# gc.collect()
# torch.cuda.synchronize()
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# import torch
# from vllm import LLM, SamplingParams
# for i in range(torch.cuda.device_count()):
#     print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
#     print(f"Memory Usage:")
#     print(f"Allocated: {torch.cuda.memory_allocated(i)/1024**3} GB")
#     print(f"Cached:    {torch.cuda.memory_reserved(i)/1024**3} GB")

# with torch.cuda.device(1):
#     #print("The cuda visible devices are: ", os.environ["CUDA_VISIBLE_DEVICES"])
#     model2 = LLM(model=cfg.full_model_name, tokenizer=cfg.full_model_name, seed=cfg.seed, tensor_parallel_size=0.5, device=f'cuda:1') #, max_num_batched_tokens=65528, max_model_len=65528)
# print("placed model 2")      
# print(f"cuda memory allocated: {torch.cuda.memory_allocated() // 1024 // 1024}MB")  
