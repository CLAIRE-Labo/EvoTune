import numpy as np
import wandb
import copy

from packing.evaluate.registry import TASK_REGISTRY


def programs_to_prompt_creative(cfg, programs, scores):
    """
    Given a list of programs and their scores, construct a prompt that encourages creative solutions.
    """
    main_txt = ""
    for program, score in zip(programs, scores):
        main_txt += f"{program}\n"
        main_txt += f"# Score achieved with the function above: {score}\n\n"

    task = TASK_REGISTRY.get(cfg.task.task_name)
    system_prompt = task["system_prompt"]
    append_prompt = task["append_prompt"]

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
    programs, scores, island_id_prompt, temperature, probabilities = programdatabase.get_prompt(
        percentile=current_percentile)

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

    return (
        chats_batch,
        island_id_prompt_batch,
        prompt_scores_batch,
        temperatures_batch,
        probabilities_batch,
        prompt_nums_batch,
        running_dict,
    )
