import numpy as np
import wandb
import logging
import copy
import os
from packing.logging.logging import wandb_log_plot, write_to_json

append_prompt_code = """You are tasked with creating a new function update_matrix() that outperforms the other two presented functions. To achieve this, follow these guidelines:
Be innovative: avoid simply rewriting or rephrasing existing approaches. Instead, use your extensive knowledge of linear algebra, optimization techniques, and mathematical principles to propose a novel approach.
Analyze the score drivers: analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
To summarize, your task is to write a new function named update_matrix() that will perform better than both functions above and achieve a higher score."""

append_prompt_bin = """You are tasked with creating a new function, priority(), that outperforms the other two presented functions. To achieve this, follow these guidelines:
Think Outside the Box: Avoid simply rewriting or rephrasing existing approaches. Prioritize creating novel solutions rather than making superficial tweaks.
Analyze the Score Drivers: Analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
Experiment with Variations: Use the insights to create a new function that builds upon successful ideas but introduces innovative variations. Consider entirely new strategies or optimizations that were not present in the previous attempts.
To summarize, your task is to write a new function named priority() that will perform better than both functions above and achieve a higher score."""

append_prompt_tsp = """You are tasked with creating a new function, heuristics(), that outperforms the other two presented functions. 
The heuristics() function takes as input a distance matrix, and returns prior indicators of how undesirable it is to include each edge in a solution. The returned matrix should be of the same shape as the input. 
When writing the new function, follow these guidelines:
Think Outside the Box: Avoid simply rewriting or rephrasing existing approaches. Prioritize creating novel solutions rather than making superficial tweaks.
Analyze the Score Drivers: Analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
To summarize, your task is to write a new function named heuristics() that will perform better than both functions above and achieve a higher score."""

append_prompt_flatpack = """You are tasked with creating a new function, priority(), that outperforms the other two presented functions. 
The priority() function takes three inputs:
1. current_grid: numpy array (float32) of shape (num_rows, num_cols) with values in the range [0, num_blocks] (corresponding to the number of each block). This grid will have zeros where no blocks have been placed and numbers corresponding to each block where that particular block has been placed.
2. blocks: numpy array (float32) of shape (num_blocks, 3, 3) of all possible blocks in that can fit in the current grid. These blocks will always have shape (3, 3).
3. action_mask: numpy array (bool) of shape (num_blocks, 4, num_rows-2, num_cols-2), representing which actions are possible given the current state of the grid. The first index indicates the block index, the second index indicates the rotation index, and the third and fourth indices indicate the row and column coordinate of where a blocks top left-most corner may be placed respectively. These values will always be num_rows-2 and num_cols-2 respectively to make it impossible to place a block outside the current grid.
It returns a numpy array of size (num_blocks, 4, num_rows-2, num_cols-2) representing how valuable it is to place a block with a rotation with its top-left corner on the row,col position in the grid.
When writing the new function, follow these guidelines:
Think Outside the Box: Avoid simply rewriting or rephrasing existing approaches. Prioritize creating novel solutions rather than making superficial tweaks.
Analyze the Score Drivers: Analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
To summarize, your task is to write a new function named priority() that will perform better than both functions above and achieve a higher score."""

append_prompt_jssp = """You are tasked with creating a new function, priority(), that outperforms the other two presented functions. 
The priority() function takes six inputs:
1. ops_machine_ids: Numpy array (int32) of shape (num_jobs, max_num_ops). For each job, it specifies the machine each op must be processed on. Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
2. ops_durations: Numpy array (int32) of shape (num_jobs, max_num_ops). For each job, it specifies the processing time of each operation. Note that a -1 corresponds to padded ops since not all jobs have the same number of ops.
3. ops_mask: Numpy array (bool) of shape (num_jobs, max_num_ops). For each job, indicates which operations remain to be scheduled. False if the op has been scheduled or if the op was added for padding, True otherwise. The first True in each row (i.e. each job) identifies the next operation for that job.
4. machines_job_ids: Numpy array (int32) of shape (num_machines,). For each machine, it specifies the job currently being processed. Note that -1 means no-op in which case the remaining time until available is always 0.
5. machines_remaining_times: Numpy array (int32) of shape (num_machines,). For each machine, it specifies the number of time steps until available.
6. action_mask: Numpy array (bool) of (num_machines, num_jobs + 1). For each machine, it indicates which jobs (or no-op) can legally be scheduled. The last column corresponds to no-op.
It returns a 1D numpy array of shape (num_jobs,) with priority scores for each job, also representing the value. Jobs with a higher score will be prioritized to be scheduled first on their machine
When writing the new function, follow these guidelines:
Think Outside the Box: Avoid simply rewriting or rephrasing existing approaches. Prioritize creating novel solutions rather than making superficial tweaks.
Analyze the Score Drivers: Analyze the characteristics of the higher-scoring function. Identify what it is doing differently or more effectively than the lower-scoring function. Determine which specific changes or techniques lead to better performance.
To summarize, your task is to write a new function named priority() that will perform better than both functions above and achieve a higher score."""

system_prompt_code = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."
system_prompt_bin = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."
system_prompt_tsp = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."
system_prompt_flatpack = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions, and you pay attention to whether shapes match."
system_prompt_jssp = "You are helpful, excellent and innovative problem solver specializing in mathematical optimization and algorithm design. You are an expert in writing Python functions."


def programs_to_prompt_creative(cfg, programs, scores):
    """
    Given a list of programs and their scores, construct a prompt that encourages creative solutions.
    """
    main_txt = ""
    for program, score in zip(programs, scores):
        main_txt += f"{program}\n"
        main_txt += f"# Score achieved with the function above: {score}\n\n"

    if cfg.task_code:
        system_prompt = system_prompt_code
        append_prompt = append_prompt_code
    elif cfg.task_bin:
        system_prompt = system_prompt_bin
        append_prompt = append_prompt_bin
    elif cfg.task_tsp:
        system_prompt = system_prompt_tsp
        append_prompt = append_prompt_tsp
    elif cfg.task_flatpack:
        system_prompt = system_prompt_flatpack
        append_prompt = append_prompt_flatpack
    elif cfg.task_jssp:
        system_prompt = system_prompt_jssp
        append_prompt = append_prompt_jssp
    else:
        raise ValueError("Task not recognized")
    chat = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{main_txt}\n{append_prompt}",
        },
    ]
    return chat, scores


def programs_to_prompt(cfg, programs, scores):
    """
    Given a list of programs and their scores, construct a prompt.
    """

    if cfg.descending_order:
        programs = programs[::-1]
        scores = scores[::-1]

    main_txt = ""
    for program, score in zip(programs, scores):
        main_txt += f"{program}\n"
        main_txt += f"# Score achieved with the function above: {score}\n\n"

    chat = [
        {
            "role": "user",
            "content": f"{main_txt}\nWrite a new function {cfg.function_str_to_extract}() that will perform better than both functions above and achieve a higher score.",
        }
    ]
    return chat, scores


def generate_prompt(cfg, programdatabase, current_percentile):
    """
    Generates a single prompt.
    """
    # Sample from the programbank
    programs, scores, island_id_prompt, temperature, probabilities = programdatabase.get_prompt(percentile=current_percentile)

    # Construct prompt
    if cfg.creative_prompt:
        chat, prompt_scores = programs_to_prompt_creative(cfg, programs, scores)
    else:
        chat, prompt_scores = programs_to_prompt(cfg, programs, scores)
    return chat, prompt_scores, island_id_prompt, temperature, probabilities


def generate_batch_prompts(cfg, programdatabase, running_dict, round_num):
    """
    Generates cfg.num_cont_rounds prompts that will be used by the producer. 
    """
    chats_batch = []
    island_id_prompt_batch = []
    prompt_scores_batch = []
    temperatures_batch = []
    probabilities_batch = []
    prompt_nums_batch = []
    # Linear decrease in the percentile based on how far we are in the rounds
    initial_percentile = cfg.initial_percentile
    final_percentile = cfg.final_percentile
    current_percentile = initial_percentile - (initial_percentile - final_percentile) * round_num / cfg.num_rounds
    running_dict["current_percentile"] = current_percentile

    for _ in range(cfg.num_cont_rounds):
        chat, prompt_scores, island_id_prompt, temperature, probabilities = generate_prompt(
            cfg, programdatabase, current_percentile
        )
        chats_batch.append(chat)
        island_id_prompt_batch.append(island_id_prompt)
        prompt_scores_batch.append(prompt_scores)
        temperatures_batch.append(temperature)
        probabilities_batch.append(probabilities)
        running_dict["prompt_num"] += 1
        prompt_nums_batch.append(copy.deepcopy(running_dict["prompt_num"]))
    
    # if cfg.wandb:
    #     island_program_scores_list = []
    #     island_sampled_program_scores = {key: [] for key in range(programdatabase.num_islands)}
    #     island_sampled_program_scores_list = np.array([score for score_list in prompt_scores_batch for score in score_list])
    #     island_id_prompt_batch_concat = np.array([int(index) for j, index in enumerate(island_id_prompt_batch) for _ in range(len(prompt_scores_batch[j]))])

    #     for i in range(programdatabase.num_islands):
    #         island_program_scores = np.array(list(programdatabase._islands[i]._clusters.keys()))
    #         island_program_scores_list.extend(island_program_scores)
    #         t = -700.0
    #         scores = [[s] for s in island_program_scores if s > t]
    #         json_file_path = f"{cfg.logs_dir}/island_program_scores_{i}.json"
    #         write_to_json(json_file_path, {"round_num": running_dict['round_num'], "scores": island_program_scores.tolist()})

    #         # write to file only the scores above some threshold t
    #         json_file_path = f"{cfg.logs_dir}/island_program_scores_t_{i}.json"
    #         write_to_json(json_file_path, {"round_num": running_dict['round_num'], "threshold": t, "scores": [score for score in island_program_scores if score > t]})

    #         table = wandb.Table(data=scores, columns=["scores"])
    #         histogram_title = f"Island {i} program scores"
    #         wandb_key = f"Island {i}/program scores"
    #         wandb.log(
    #                 {
    #                     "round_num_hist": copy.deepcopy(round_num),
    #                     wandb_key: wandb.plot.histogram(table, "scores", title=histogram_title),
    #                 }
    #             )
            
    #         sampled_program_scores = island_sampled_program_scores_list[island_id_prompt_batch_concat == i]
    #         if len(list(sampled_program_scores)) == 0:
    #             continue
    #         try:
    #             island_sampled_program_scores[i].extend(sampled_program_scores)
    #         except:
    #             island_sampled_program_scores[i].append(sampled_program_scores)
    #         if len(island_sampled_program_scores[i]) == 0:
    #             continue

    #         scores = [[s] for s in island_sampled_program_scores[i] if s > t]
    #         write_to_json(f"{cfg.logs_dir}/island_sampled_program_scores_{i}.json", {"round_num": running_dict['round_num'], "scores": island_sampled_program_scores[i]})
    #         write_to_json(f"{cfg.logs_dir}/island_sampled_program_scores_t_{i}.json", {"round_num": running_dict['round_num'], "threshold": t, "scores": [s for s in island_sampled_program_scores[i] if s > t]})
    #         write_to_json(f"{cfg.logs_dir}/scores_{i}.json", {"round_num": running_dict['round_num'], "threshold": t, "scores": scores})
    #         table = wandb.Table(data=scores, columns=["scores"])
    #         histogram_title = f"Sampled programs scores for island {i}"
    #         wandb_key = f"Island {i}/sampled program scores"
    #         wandb.log(
    #                 {
    #                     "round_num_hist": copy.deepcopy(round_num),
    #                     wandb_key: wandb.plot.histogram(table, "scores", title=histogram_title),
    #                 }
    #             )

    #     # only write every 200 rounds
    #     # if running_dict['round_num'] % 200 == 0:
    #     write_to_json(f"{cfg.logs_dir}/programdb_scores.json", {"round_num": running_dict['round_num'], "scores": island_program_scores_list})
    #     write_to_json(f"{cfg.logs_dir}/programdb_scores_t.json", {"round_num": running_dict['round_num'], "scores": [score for score in island_program_scores_list if score > t]})
    #     metrics = {}
    #     all_scores = island_program_scores_list
    #     best_10_scores = np.sort(all_scores)[::-1][:10]
    #     best_10_scores_avg_overall = np.mean(best_10_scores)
    #     best_50_scores = np.sort(all_scores)[::-1][:50]
    #     best_50_scores_avg_overall = np.mean(best_50_scores)
    #     metrics["round_num"] = round_num
    #     metrics["evalset"] = "train"
    #     metrics["all_scores"] = all_scores
    #     metrics["best_overall_score"] = np.max(all_scores)
    #     metrics["best_10_scores_avg_overall"] = best_10_scores_avg_overall
    #     metrics["best_50_scores_avg_overall"] = best_50_scores_avg_overall
    #     metrics["num_unique_scores"] = len(set(all_scores))
    #     metrics["score_threshold_best_1_percent"] = np.percentile(all_scores, 99)
    #     metrics["score_threshold_best_10_percent"] = np.percentile(all_scores, 90)
    #     metrics["score_threshold_best_20_percent"] = np.percentile(all_scores, 80)
    #     metrics["score_threshold_best_40_percent"] = np.percentile(all_scores, 60)
    #     metrics["time_taken_to_evaluate"] = 0.0
    #     # create a metrics folder if it does not exist
    #     if not os.path.exists(f"{cfg.logs_dir}/metrics"):
    #         os.makedirs(f"{cfg.logs_dir}/metrics")
    #     write_to_json(f"{cfg.logs_dir}/metrics/metrics_train_round_{round_num}.json", metrics)
    #     # log to wandb
    #     for binsize in [5, 10, 20]:
    #         wandb_log_plot(cfg, all_scores, f"trainset/pdb_hist_bsize_{binsize}", f"All programs score distribution (round {round_num})", "Program Scores", "Frequency", plot_type='hist', binsize=binsize, low=int(t), high=-150)
    #     del island_program_scores_list
    #     del island_sampled_program_scores
    #     del island_sampled_program_scores_list
    #     del island_id_prompt_batch_concat
    #     del island_program_scores
        
        # for i in range(cfg.num_cont_rounds):
        #     logging.info(f"Prompt {i}, Island {island_id_prompt_batch[i]} program scores: {prompt_scores_batch[i]}")
        #     scores = [[s] for s in prompt_scores_batch[i]]
        #     table = wandb.Table(data=scores, columns=["scores"])
        #     wandb.log(
        #         {
        #             "round_num_hist": f"{running_dict['round_num']}",
        #             f"Island {island_id_prompt_batch[i]}/Prompt scores at round {i}": wandb.plot.histogram(table, "scores", title="Sampled programs scores"),
        #         }
        #     )

    return (
        chats_batch,
        island_id_prompt_batch,
        prompt_scores_batch,
        temperatures_batch,
        probabilities_batch,
        prompt_nums_batch,
        running_dict,
    )
