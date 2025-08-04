import math

def learning_rate_schedule(cfg, lr, num_datapoints, tuning_loop, round_num):
    print(f"Learning rate before schedule: {lr}")
    lr = lr * math.sqrt(1000 / round_num) 

    if cfg.lr_annealing:
        lr = lr / (2**tuning_loop)

    assert lr > 0, "Learning rate is zero"
    print(f"Learning rate after schedule: {lr}")

    return lr