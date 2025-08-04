from dataclasses import is_dataclass, fields
from numpy import ndarray
import wandb
import numpy as np
from omegaconf import DictConfig, OmegaConf
import copy
from packing.logging.log_to_file import save_func_class_to_file, save_failed_func_class_to_file, write_chat_to_file
import math
from packing.logging.function_class import FunctionClass
import logging
import matplotlib.pyplot as plt
import os
import json


def wandb_log_running_dict(running_dict):
    # If one of the logged values is -inf, log it as Nan
    wandb_dict = copy.deepcopy(running_dict)
    # Filter out all arrays
    for key, value in running_dict.items():
        if isinstance(value, ndarray):
            continue
        else:
            if value == float("-inf") or value == float("inf"):
                wandb_dict[key] = float("nan")
    wandb.log(wandb_dict)


def log_to_wandb(function_class_instance, running_dict, prefix=""):
    # Determine what to log from running_dict
    extra_dict = {"num_func_evaluated": running_dict["num_func_evaluated"], "round_num": running_dict["round_num"]}
    # Determine what to log from running_dict
    if function_class_instance.fail_flag == 0:
        # Passed
        extra_dict["num_func_passed"] = running_dict["num_func_passed"]
        if function_class_instance.correct_flag:
            # Correct
            extra_dict["num_func_correct"] = running_dict["num_func_correct"]
    elif function_class_instance.fail_flag == 1:
        # Failed
        extra_dict["num_func_failed"] = running_dict["num_func_failed"]
    else:
        raise ValueError("Invalid fail_flag")

    # Flatten the dataclass instance to a dictionary    
    flattened_dict = {}

    def flatten(prefix, obj):
        for field in fields(obj):
            value = getattr(obj, field.name)
            key = f"{prefix}/{field.name}" if prefix else field.name

            if is_dataclass(value):
                # Recursively flatten nested dataclasses
                flatten(key, value)
            else:
                # Check if the value is acceptable for logging
                if is_loggable(value):
                    flattened_dict[key] = value

    def is_loggable(value):
        # Accept scalar types: int, float, bool, str, and NumPy scalars
        if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
            return True
        else:
            return False  # Exclude non-scalar types

    flatten(prefix, function_class_instance)

    if prefix:
        prefixed_extra_dict = {f"{prefix}/{k}": v for k, v in extra_dict.items()}
    else:
        prefixed_extra_dict = extra_dict

    flattened_dict.update(prefixed_extra_dict)

    # Log the flattened dictionary to wandb
    wandb.log(flattened_dict)


def log_train_keys_containing_loop(cfg, running_dict):
    if cfg.multiple_models:
        assert all(
            [
                running_dict[f"{model_name}_loop_num"]
                for model_name in cfg.model.model_name
            ]
        )
        running_dict[f"loop_num"] = copy.deepcopy(
            running_dict[f"{cfg.model.model_name[0]}_loop_num"]
        )

    else:
        running_dict[f"loop_num"] = copy.deepcopy(
            running_dict[f"{cfg.model.model_name}_loop_num"]
        )

    # Find all keys in running dict that have the word "loop" in them
    keys_with_loop = [
        key for key in running_dict.keys() if "traindata_loop" in key
    ]
    data_with_loop = {key: running_dict[key] for key in keys_with_loop}

    for idx, loop_num in enumerate(running_dict[f"loop_num"]):
        # Collect first elements of data_with_loop and log them
        for key, value in data_with_loop.items():
            logging.info(
                f"Logging {key} with value {value[idx]} and loop_num {loop_num}"
            )
            wandb.log({key: value[idx], "loop_num": loop_num})

    # Delete keys_with_loop from running_dict
    for key in keys_with_loop:
        del running_dict[key]

    return running_dict


def initialize_running_dict(cfg, input_struct):
    running_dict = {
        "weight_distributions": {},
        "canonical_forms": {},
        "best_true_score_from_score": - math.inf,
        "best_true_score": - math.inf,
        "best_score": - math.inf,
        "num_func_generated": 0,
        "num_func_evaluated": 0,
        "num_func_correct": 0,
        "num_func_failed": 0,
        "num_func_passed": 0,
        "prompt_num": 0,
        "round_num": -1,
        "flag_load_finetuned": 0,
    }

    return running_dict


def register_programs(
        cfg,
        running_dict,
        passed_classes,
        failed_classes,
        programdatabase,
):
    print(f"\nNumber of passing functions to register: {len(passed_classes)}\n")
    print(f"\nNumber of failed functions: {len(failed_classes)}\n")

    flag_writeintxt = False  # In case results is empty
    flag_improvement = False

    for passed_class in passed_classes:
        assert isinstance(passed_class, FunctionClass)
        assert passed_class.fail_flag == 0

        running_dict["num_func_evaluated"] += 1
        running_dict["num_func_passed"] += 1

        # Only register the program in the program bank if it is correct 
        if passed_class.correct_flag:
            running_dict["num_func_correct"] += 1

            # Only do this for correct functions
            if passed_class.score > programdatabase.get_best_score:
                running_dict["best_score"] = passed_class.score
                running_dict["best_true_score_from_score"] = passed_class.true_score
                flag_improvement = True

            if passed_class.true_score > running_dict["best_true_score"]:
                running_dict["best_true_score"] = passed_class.true_score

            # Register the program in the program bank
            if passed_class.imports_str == "":
                program_to_register = passed_class.function_str
            else:
                program_to_register = passed_class.imports_str + "\n" + passed_class.function_str

            # Register after checking for improvement
            flag_potential_change = programdatabase.register_program(
                program_to_register, passed_class.score, passed_class.prompt.island_id
            )
            # Change write_to_txt flag as True if one of write_to_txt or flag_potential_change is True
            flag_writeintxt = flag_writeintxt or flag_potential_change

            if flag_writeintxt:
                save_func_class_to_file(cfg, passed_class)

        # Needs to be here to log the correct number of functions
        if cfg.wandb:
            log_to_wandb(passed_class, running_dict)

        # Write the chat to a file
        write_chat_to_file(cfg, running_dict, passed_class, folder_name="chats")

    running_dict["flag_improvement"] = int(flag_improvement)

    for failed_class in failed_classes:
        running_dict["num_func_evaluated"] += 1
        running_dict["num_func_failed"] += 1
        # Needs to be here to log the correct number of functions
        if cfg.wandb:
            log_to_wandb(failed_class, running_dict)

        write_chat_to_file(cfg, running_dict, failed_class, folder_name="failed_chats")

        save_failed_func_class_to_file(cfg, failed_class)

    return programdatabase, running_dict, flag_writeintxt


def get_pd_statistics(cfg, programdatabase, running_dict, round_num):
    # Log the statistics
    for island_id in range(len(programdatabase._islands)):
        island = programdatabase._islands[island_id]
        best_island_score = programdatabase.get_best_score_per_island(island_id)
        running_dict[f"islands_scores/best_program_score_in_island_{island_id}"] = best_island_score
        running_dict[f"islands_clusters/num_clusters_in_island_{island_id}"] = len(island._clusters)

        # Get the number of programs in clusters in the island
        num_of_programs_per_cluster_in_island = []
        for key in island._clusters.keys():
            num_of_programs_per_cluster_in_island.append(len(island._clusters[key]._programs))

        running_dict[f"clusters_avg/average_num_programs_per_cluster_in_island_{island_id}"] = np.mean(
            num_of_programs_per_cluster_in_island
        )
        running_dict[f"clusters_total/total_num_programs_in_island_{island_id}"] = np.sum(
            num_of_programs_per_cluster_in_island
        )

    running_dict["best_overall_score"] = programdatabase.get_best_score
    running_dict["num_func/num_functions_in_programbank"] = programdatabase.total_num_programs

    # Add less noisy evaluation metric
    island_program_scores_list = []

    for i in range(programdatabase.num_islands):
        island_program_scores = np.array(list(programdatabase._islands[i]._clusters.keys()))
        island_program_scores_list.append(island_program_scores)

    # Get best 10 scores from each island
    best_10_scores_per_island = []
    for island_program_scores in island_program_scores_list:
        best_10_scores_per_island.extend(np.sort(island_program_scores)[::-1][:10])
    best_10_scores_avg_across_islands = np.mean(best_10_scores_per_island)

    # Get best 100 scores from each island  
    best_100_scores_per_island = []
    for island_program_scores in island_program_scores_list:
        best_100_scores_per_island.extend(np.sort(island_program_scores)[::-1][:100])
    best_100_scores_avg_across_islands = np.mean(best_100_scores_per_island)

    # Get overall best 10 and 50 scores
    # Flatten the island_program_scores_list
    all_scores = []
    for island_program_scores in island_program_scores_list:
        all_scores.extend(island_program_scores)
    all_scores = np.array(all_scores)

    best_10_scores = np.sort(all_scores)[::-1][:10]
    best_10_scores_avg_overall = np.mean(best_10_scores)

    best_50_scores = np.sort(all_scores)[::-1][:50]
    best_50_scores_avg_overall = np.mean(best_50_scores)

    running_dict["Best 10 scores avg across islands"] = best_10_scores_avg_across_islands
    running_dict["Best 100 scores avg across islands"] = best_100_scores_avg_across_islands
    running_dict["Best 10 scores overall"] = best_10_scores_avg_overall
    running_dict["Best 50 scores overall"] = best_50_scores_avg_overall

    running_dict["Number of unique scores in program bank"] = len(set(all_scores))
    running_dict[r"trainset/Score threshold of best 1% of programs"] = np.percentile(all_scores, 99)
    running_dict[r"trainset/Score threshold of best 10% of programs"] = np.percentile(all_scores, 90)
    running_dict[r"trainset/Score threshold of best 20% of programs"] = np.percentile(all_scores, 80)
    running_dict[r"trainset/Score threshold of best 40% of programs"] = np.percentile(all_scores, 60)

    if cfg.wandb:
        island_program_scores_list = []

        for i in range(programdatabase.num_islands):
            island_program_scores = np.array(list(programdatabase._islands[i]._clusters.keys()))
            island_program_scores_list.extend(island_program_scores)
            t = -700.0
            scores = [[s] for s in island_program_scores if s > t]
            # create a directory for island_program_scores if it does not exist
            if not os.path.exists(f"{cfg.logs_dir}/island_scores"):
                os.makedirs(f"{cfg.logs_dir}/island_scores")
            json_file_path = f"{cfg.logs_dir}/island_scores/island_program_scores_{i}.json"
            write_to_json(json_file_path, {"round_num": round_num, "scores": island_program_scores.tolist()})

            # write to file only the scores above some threshold t
            json_file_path = f"{cfg.logs_dir}/island_scores/island_program_scores_t_{i}.json"
            write_to_json(json_file_path, {"round_num": round_num, "threshold": t,
                                           "scores": [score for score in island_program_scores if score > t]})

        # if running_dict['round_num'] % 200 == 0:
        write_to_json(f"{cfg.logs_dir}/programdb_scores.json",
                      {"round_num": round_num, "scores": island_program_scores_list})
        write_to_json(f"{cfg.logs_dir}/programdb_scores_t.json",
                      {"round_num": round_num, "scores": [score for score in island_program_scores_list if score > t]})
        metrics = {}
        all_scores = island_program_scores_list
        best_10_scores = np.sort(all_scores)[::-1][:10]
        best_10_scores_avg_overall = np.mean(best_10_scores)
        best_50_scores = np.sort(all_scores)[::-1][:50]
        best_50_scores_avg_overall = np.mean(best_50_scores)
        metrics["round_num"] = round_num
        metrics["evalset"] = "train"
        metrics["all_scores"] = all_scores
        metrics["best_overall_score"] = np.max(all_scores)
        metrics["best_10_scores_avg_overall"] = best_10_scores_avg_overall
        metrics["best_50_scores_avg_overall"] = best_50_scores_avg_overall
        metrics["num_unique_scores"] = len(set(all_scores))
        metrics["score_threshold_best_1_percent"] = np.percentile(all_scores, 99)
        metrics["score_threshold_best_10_percent"] = np.percentile(all_scores, 90)
        metrics["score_threshold_best_20_percent"] = np.percentile(all_scores, 80)
        metrics["score_threshold_best_40_percent"] = np.percentile(all_scores, 60)
        metrics["time_taken_to_evaluate"] = 0.0
        # create a metrics folder if it does not exist
        if not os.path.exists(f"{cfg.logs_dir}/metrics"):
            os.makedirs(f"{cfg.logs_dir}/metrics")
        write_to_json(f"{cfg.logs_dir}/metrics/metrics_train_round_{round_num}.json", metrics)

        del island_program_scores_list
        del island_program_scores

    # del island_program_scores_list
    del best_10_scores_per_island
    del best_100_scores_per_island
    del all_scores
    del best_10_scores
    del best_50_scores

    return running_dict


def calculate_avg_passed_score(running_dict, passed_function_classes):
    batch_scores = []
    batch_true_scores = []
    for passed_function_class in passed_function_classes:
        batch_scores.append(passed_function_class.score)
        batch_true_scores.append(passed_function_class.true_score)
    avg_passed_score = np.mean(batch_scores)
    avg_passed_true_score = np.mean(batch_true_scores)

    running_dict["avg_passed_score"] = avg_passed_score
    running_dict["avg_passed_true_score"] = avg_passed_true_score
    return running_dict


def calculate_avg_score(running_dict, passed_function_classes, failed_function_classes):
    batch_scores = []
    batch_true_scores = []
    for func_class in passed_function_classes + failed_function_classes:
        batch_scores.append(func_class.score)
        batch_true_scores.append(func_class.true_score)
    avg_score = np.mean(batch_scores)
    avg_true_score = np.mean(batch_true_scores)

    running_dict["avg_score"] = avg_score
    running_dict["avg_true_score"] = avg_true_score
    return running_dict


def wandb_log_plot(cfg, data, wandb_name, title, xlabel, ylabel, plot_type='hist', **kwargs):
    plt.figure(figsize=(10, 8), dpi=100)
    if plot_type == 'hist':
        binsize = kwargs.get('binsize', 10)
        low = kwargs.get('low', -700)
        high = kwargs.get('high', -150)
        plt.hist(data, bins=list(range(low, high, binsize)), edgecolor='black', color='#1f77b4')
        plt.xticks(list(range(low, high, 40)))
    elif plot_type == 'scatter':
        low = kwargs.get('low', -700)
        x, y = data[0], data[1]
        y = [y[i] for i in range(len(y)) if x[i] > low]
        x = [x[i] for i in range(len(x)) if x[i] > low]
        plt.scatter(x, y)
        d = (max(x) - min(x)) / 10.
        plt.plot([min(x) - d, max(x) + d], [min(x) - d, max(x) + d], color='red')
        plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # log the plot as an artifact to wandb
    wandb.log({wandb_name: wandb.Image(plt)})


def write_to_json(json_file, data):
    if os.path.exists(json_file):
        with open(json_file, "r") as f:
            try:
                all_data = json.load(f)
            except json.JSONDecodeError:
                os.remove(json_file)
                all_data = []
    else:
        all_data = []
    all_data.append(data)
    with open(json_file, "w") as f:
        json.dump(all_data, f, indent=2)
    return
