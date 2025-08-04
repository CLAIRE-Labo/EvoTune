import csv

seeds = range(0, 10)
train_types = ['dpo', 'none']
models = ['llama32', 'phi', 'granite']
learning_rates = [1e-5, 1e-4]
task = ['flatpack']

if __name__ == '__main__':
    combinations = [
        dict(seed=seed, train_type=train, model_name=model, learning_rate=learning_rate)
        for model in models
        for train in train_types
        for seed in seeds
        for learning_rate in learning_rates
    ]

    with open("sweep_config.csv", "w", newline="\n") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["seed", "train", "model", "train.dpo_config.learning_rate"])
        for combo in combinations:
            writer.writerow([combo["seed"], combo["train_type"], combo["model_name"], combo["learning_rate"]])
