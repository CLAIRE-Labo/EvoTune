
import pickle
import json
import copy
from datetime import datetime
from dataclasses import asdict
from packing.logging.function_class import FunctionClass
import numpy as np
import os

def serialize(function_class):
    def _convert(obj):
        if isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.astype(int).tolist()
        elif isinstance(obj, list):
            return [_convert(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: _convert(value) for key, value in obj.items()}
        elif hasattr(obj, "serialize") and callable(getattr(obj, "serialize")):
            # If the object has its own serialize method, use it
            return obj.serialize()
        elif hasattr(obj, "__dict__"):
            # Convert objects with a __dict__ attribute (like custom classes) to dict
            return _convert(asdict(obj))
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # Fallback: convert to string
            return str(obj)

    eval_dict = asdict(function_class)
    return _convert(eval_dict)

def save_func_class_to_file(cfg, result_function_class: FunctionClass, filename="best_programs.txt"):
    print("Saving a program to a file...")
    with open(f"{cfg.logs_dir}/{filename}", "a") as f:
        f.write("\n\n")
        f.write(f"# Score: {result_function_class.score}\n")
        f.write(f"# True score {result_function_class.true_score}\n")
        # Append the best programs to the txt file
        if cfg.task_code:
            f.write(f"n = {cfg.n}\n")
            f.write(f"k = {cfg.k}\n")
        f.write(result_function_class.imports_str)
        f.write("\n")
        f.write(result_function_class.function_str)
        f.write("\n\n")
        # Changes for JSON serialization
        eval_dict = serialize(result_function_class)
        f.write(json.dumps(eval_dict))

def save_failed_func_class_to_file(cfg, result_function_class: FunctionClass, filename="failed_programs.txt"):
    with open(f"{cfg.logs_dir}/{filename}", "a") as f:
        f.write("\n\n")
        f.write(f"# Score: {result_function_class.score}\n")
        f.write(result_function_class.imports_str)
        f.write("\n")
        f.write(result_function_class.function_str)
        f.write("\n\n")
        f.write(f"Fail reason: {result_function_class.identify_fail_reason()}")
        f.write("\n\n\n\n")

def write_chat_to_file(cfg, running_dict, function_class: FunctionClass, folder_name="chats"):
    with open(f"{cfg.logs_dir}/{folder_name}/{running_dict['num_func_evaluated']}.txt", "w") as f:
        f.write(f"# Score: {function_class.score}\n")
        for chat_write in function_class.chat:
            f.write(chat_write["role"])
            f.write("\n\n")
            f.write(chat_write["content"])
            f.write("\n\n\n\n")
        if function_class.fail_flag:
            f.write(f"# Failed reason: {function_class.identify_fail_reason()}")


def save_pd_to_file(cfg, programdatabase, round_num):
    print("Saving the programs from program bank to a file...")
    time_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    with open(f"{cfg.logs_dir}/programbank/{time_now}.txt", "w") as f:
        for island_id in range(len(programdatabase._islands)):
            program_str = programdatabase._best_program_per_island[island_id]
            if program_str is not None:
                score = programdatabase._best_score_per_island[island_id]
                if cfg.task_code:
                    f.write(f"# Score: {score}\n")
                    f.write(f"n = {cfg.n}\n")
                    f.write(f"k = {cfg.k}\n")
                    f.write(program_str)
                    f.write("\n\n")
                else:
                    f.write(f"# Score: {score}\n")
                    f.write(f"n = {cfg.n}\n")
                    f.write(f"d = {cfg.dist}\n")
                    f.write(f"keep_perc = {cfg.keep_perc}\n\n")
                    f.write(program_str)
                    f.write("\n\n")


def save_in_logsdir(
    logs_dir,
    running_dict,
    programdatabase,
    sft_chats,
    dpo_chats,
    round_num,
    input_struct,
):
    # Save the pickled program bank
    with open(f"{logs_dir}/programbank.pkl", "wb") as f:
        pickle.dump(programdatabase, f)

    with open(f"{logs_dir}/programbank_round{round_num}.pkl", "wb") as f:
        pickle.dump(programdatabase, f)

    # Save the finetuning_data_chats
    with open(f"{logs_dir}/sft_chats.pkl", "wb") as f:
        pickle.dump(sft_chats, f)

    # Save the dpo_data_chats
    with open(f"{logs_dir}/dpo_chats.pkl", "wb") as f:
        pickle.dump(dpo_chats, f)

    # Set generated functions to evaluated functions in case the run is stopped
    #running_dict["num_func_generated"] = copy.deepcopy(running_dict["num_func_evaluated"])
    with open(f"{logs_dir}/running_dict.pkl", "wb") as f:
        pickle.dump(running_dict, f)

    # Save num_rounds as a number in a text file
    with open(f"{logs_dir}/round_num.txt", "w") as f:
        f.write(str(round_num))

    with open(f"{logs_dir}/input_struct.pkl", "wb") as f:
        pickle.dump(input_struct, f)


def load_from_logsdir(logs_dir):
    with open(f"{logs_dir}/programbank.pkl", "rb") as f:
        programdatabase = pickle.load(f)

    with open(f"{logs_dir}/sft_chats.pkl", "rb") as f:
        sft_chats = pickle.load(f)

    with open(f"{logs_dir}/dpo_chats.pkl", "rb") as f:
        dpo_chats = pickle.load(f)

    with open(f"{logs_dir}/running_dict.pkl", "rb") as f:
        running_dict = pickle.load(f)
    #assert running_dict["num_func_generated"] == running_dict["num_func_evaluated"]

    with open(f"{logs_dir}/round_num.txt", "r") as f:
        starting_round = int(f.read())
    
    with open(f"{logs_dir}/input_struct.pkl", "rb") as f:
        input_struct = pickle.load(f)

    return programdatabase, sft_chats, dpo_chats, running_dict, starting_round, input_struct

def load_from_logsdir_eval(logs_dir):

    pdb_list = []

    # look for all files in logs_dir that are programbank_round*.pkl and extract the round_nums as well as the programdatabases
    for file in os.listdir(logs_dir):
        if file.startswith("programbank_round") and file.endswith(".pkl"):
            round_num = int(file.split("programbank_round")[1].split(".pkl")[0])
            with open(f"{logs_dir}/{file}", "rb") as f:
                pdb = pickle.load(f)
            pdb_list.append((round_num, pdb))
    
    # Removed to make sure we are evaluating the right round
    # # if there are no programbank_round*.pkl files, load the programbank.pkl file
    # if not pdb_list:
    #     with open(f"{logs_dir}/round_num.txt", "rb") as f:
    #         round_num = int(f.read())
    #     with open(f"{logs_dir}/programbank.pkl", "rb") as f:
    #         pdb = pickle.load(f)
    #     pdb_list.append((round_num, pdb))

    # if there are still no programbank files, raise an error
    if not pdb_list:
        raise FileNotFoundError("No programbank files found in the logs directory")

    with open(f"{logs_dir}/running_dict.pkl", "rb") as f:
        running_dict = pickle.load(f)
    #assert running_dict["num_func_generated"] == running_dict["num_func_evaluated"]

    with open(f"{logs_dir}/input_struct.pkl", "rb") as f:
        input_struct = pickle.load(f)

    return pdb_list, running_dict, input_struct
