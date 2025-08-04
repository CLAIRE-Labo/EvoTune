
import numpy as np


def calculate_dataset_statistics(cfg, running_dict, scores, scores_since_tuning, algo):
    # Get average score since beginning
    avg_score_since_beginning = sum(scores) / len(scores)
    running_dict[f"traindata{algo}/avg_score_since_beginning"] = avg_score_since_beginning

    # Get average score since last finetuning
    avg_score_since_tuning = sum(scores_since_tuning) / len(scores_since_tuning)
    running_dict[f"traindata{algo}/avg_score_since_tuning"] = avg_score_since_tuning


    # Get the score percentile since beginning
    score_percentile_since_beginning = np.percentile(scores, (100-cfg.percentile))
    running_dict[f"traindata{algo}/score_percentile_since_beginning"] = score_percentile_since_beginning

    # Get the score percentile since last finetuning
    score_percentile_since_tuning = np.percentile(scores_since_tuning, (100-cfg.percentile))
    running_dict[f"traindata{algo}/score_percentile_since_tuning"] = score_percentile_since_tuning


    # Get the median score since beginning
    median = np.percentile(scores, 50)
    running_dict[f"traindata{algo}/score_median_since_beginning"] = median

    # Get the median score since last finetuning
    median_since_tuning = np.percentile(scores_since_tuning, 50)
    running_dict[f"traindata{algo}/score_median_since_tuning"] = median_since_tuning

    return running_dict, score_percentile_since_tuning

