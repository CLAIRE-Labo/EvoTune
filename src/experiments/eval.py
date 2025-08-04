import multiprocessing
import os

import hydra
import numpy as np

from packing.evaluate.registry import TASK_REGISTRY

os.environ["JAX_PLATFORM_NAME"] = "cpu"  # We can't use GPU with multiprocessing for evaluation very well yet
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["WANDB_SILENT"] = "true"

import wandb
from pathlib import Path
from omegaconf import OmegaConf, ListConfig
import logging
import copy

from packing.utils.seeding import seed_everything, generate_random_seed

from packing.logging.log_to_file import (
    load_from_logsdir_eval,
)
from packing.parallel.continuous_execution import (
    evaluate_on_evalset,
)
import time

OmegaConf.register_new_resolver("generate_random_seed", generate_random_seed, use_cache=True)
OmegaConf.register_new_resolver("eval", eval, use_cache=True)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # So we can assign new fields
    cfg.wandb = 1  # Force use wandb

    evalset = cfg.evalset

    if cfg.task.task_name in {'flatpack'}:
        # Unfortunately, jax doesn't work with forked multithreading since it is multithreaded itself,
        # so we need to spawn the processes instead
        multiprocessing.set_start_method('spawn')

    if 'logs_path' in cfg and cfg.logs_path is not None:
        import os
        os.chdir(cfg.logs_path)
    if not 'logs_dir' in cfg or cfg.logs_dir is None:
        cfg.logs_dir = (
            f"out/logs/{cfg.prefix}/{cfg.task_name}_{cfg.config_common}_{cfg.config_specific}_{cfg.seed}"
        )

    # Set up logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{cfg.logs_dir}/stdout_eval.log")
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Task specific imports
    try:
        task = TASK_REGISTRY.get(cfg.task.task_name)
    except ValueError as e:
        raise RuntimeError(
            f"Task '{cfg.task.task_name}' is not registered. Check if it's implemented and registered properly.") from e

    generate_input = task["generate_input"]
    evaluate_func = task["evaluate_func"]

    # Save the config and set up wandb
    logging.info(f"Working directory: {Path.cwd()}")
    if cfg.wandb:
        default_run = wandb.init(
            project=cfg.project,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=cfg.wandb_name,
            group=cfg.group_name,
            reinit=True,
            entity=cfg.entity,
        )
        cfg.default_wandb_name = wandb.run.id

    logging.info(f"Evaluating with config: \n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # DO NOT SEED IF USING MULTIPLE COPY OF THE MODEL ON MULTIPLE GPUS, AS ALL GENERATORS WILL BE THE SAME
    seed_everything(cfg.seed)
    if cfg.multiple_models:
        assert len(set(cfg.model_name)) == len(
            cfg.model_name
        ), "Model names must be unique if using seeding, otherwise we are generating same outputs"

    logging.info(f"Evaluating from logs directory at {cfg.logs_dir}")
    programdatabase_list, running_dict, input_struct = (
        load_from_logsdir_eval(cfg.logs_dir)
    )

    # cfg.eval_frequency should be less than the maximum round_num
    # assert it's not an empty list
    if isinstance(cfg.eval_frequency, int):
        assert cfg.eval_frequency <= max(
            [x[0] for x in programdatabase_list]), "Eval frequency should be less than the maximum round number"
    elif isinstance(cfg.eval_frequency, ListConfig):
        assert len(cfg.eval_frequency) > 0, "Eval frequency should not be an empty list"
        assert max(cfg.eval_frequency) <= max(
            [x[0] for x in programdatabase_list]), "Eval frequency should be less than the maximum round number"

    for round_num, programdatabase in programdatabase_list:
        if (isinstance(cfg.eval_frequency, int) and (round_num == cfg.eval_frequency)) or (
                isinstance(cfg.eval_frequency, ListConfig) and (round_num in cfg.eval_frequency)):
            logging.info(f"Evaluating at round number {round_num}")
            logging.info(
                f"Number of functions in the programbank: {programdatabase.total_num_programs}"
            )

            time_start = time.time()
            running_dict["round_num"] = copy.deepcopy(round_num)

            input_struct_debug = copy.deepcopy(input_struct)
            logging.info("-" * 10)
            logging.info(f"EVALUATING ON EVALSET: {evalset}, ROUND NUM {round_num}")
            evaluate_on_evalset(
                cfg,
                running_dict,
                round_num,
                evaluate_func,
                programdatabase,
                evalset,
                generate_input,
            )

            assert (
                    (type(input_struct) == np.ndarray and np.all(
                        input_struct == input_struct_debug)) or input_struct == input_struct_debug
            ), "Input struct has been changed, this should not happen"

            time_end = time.time()
            logging.info(f"\n\nTime taken for one round: {int(time_end - time_start)} seconds \n\n")

    logging.info("FINISHED ALL EVALUATIONS")


if __name__ == "__main__":
    main()
