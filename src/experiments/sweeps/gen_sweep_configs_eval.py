import csv

seeds = range(0, 10)
trains = ['none', 'dpo']
# --- Chained together ---
prefixes = ['sweepfinalicml', 'sweepfinalicml', 'finalarxiv']
tasks = ['bin', 'tsp', 'flatpack']
models = ['llama32', 'granite', 'phi']
func_to_extracts = ['priority', 'heuristics', 'priority']
eval_freqs = [[100, 500]]
# ------------------------

if __name__ == '__main__':
    combinations = [
        dict(seed=seed, prefix=prefix, train=train, task=task, model=model, func_to_extract=func_to_extract,
             eval_freq=eval_freq)
        for prefix, task, func_to_extract in zip(prefixes, tasks, func_to_extracts)
        for model in models
        for train in trains
        for seed in seeds
        for eval_freq in eval_freqs
    ]

    with open("../../../configs/sweep/eval_ood_sweep_config.csv", "w", newline="\n") as f:
        writer = csv.writer(f, delimiter="|")
        # Write header
        writer.writerow(["seed", "task", "train", "model", "logs_path", "logs_dir", "wandb_name", "group_name",
                         "function_str_to_extract", "eval_frequency"])

        for combo in combinations:
            combo["logs_path"] = "/claire-rcp-scratch/shared/packing_logs/logs/"
            # This is for the old identifiers:
            ident = 'b' if combo['train'].lower() == 'none' else 's2'
            conf_tag = 'icml' if combo['task'] in {'bin', 'tsp'} else 'arxiv'
            combo["logs_dir"] = f"{combo['prefix']}/{combo['task']}_task{combo['task']}_1m{combo['model']}{conf_tag}{ident}_{combo['seed']}"
            combo["wandb_name"] = f"{combo['prefix']}/task{combo['task']}_1m{combo['model']}{conf_tag}{ident}_{combo['seed']}"
            combo["group_name"] = f"{combo['prefix']}/task{combo['task']}_1m{combo['model']}{conf_tag}{ident}"
            # ---
            writer.writerow(
                [combo["seed"], combo["task"], combo["train"], combo["model"], combo["logs_path"], combo["logs_dir"],
                 combo["wandb_name"], combo["group_name"], combo["func_to_extract"], combo["eval_freq"]])