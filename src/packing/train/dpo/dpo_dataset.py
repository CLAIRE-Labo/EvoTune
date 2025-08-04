from typing import List, Dict

from typing import List, Dict, Tuple
from collections import defaultdict
import torch
from itertools import cycle, product
import copy
import numpy as np
import wandb
from datasets import DatasetDict, Dataset


class DPODataBuffer:
    def __init__(self):
        self.train_dataset = {"prompt": [], "chosen": [], "rejected": [], "chosen_chat_score": []}
        self.scores_since_finetune = []

    def construct_and_add_pairs(self, cfg, passed_classes_previous, failed_classes_previous):
        failed_chats = []
        passed_chats = []
        passed_scores = []

        # Collect failed chats
        for failed_class in failed_classes_previous:
            assert failed_class.fail_flag == 1
            if failed_class.chat not in failed_chats:
                failed_chats.append(failed_class.chat)

        # Collect chosen chats and their scores
        for passed_class in passed_classes_previous:
            assert passed_class.fail_flag == 0
            if passed_class.chat not in passed_chats:
                passed_chats.append(passed_class.chat)
                passed_scores.append(passed_class.score)
                self.scores_since_finetune.append(passed_class.score)

        assert len(passed_chats) == len(passed_scores)

        pairs = []

        wandb.log(
            {"dpodata/passed_chats": len(passed_chats), "dpodata/failed_chats": len(failed_chats)}
        )

        if passed_chats:
            unique_scores = set(passed_scores)
            wandb.log({"dpodata/unique_passed_scores": len(unique_scores)})
            chat_ranking = sorted(
                zip(passed_chats, passed_scores), key=lambda x: x[1], reverse=True
            )
            if len(unique_scores) >= 2:
                # Sort the passed_chats and passed_scores together by scores in descending order
                # Find the highest and lowest scores
                highest_score = chat_ranking[0][1]
                lowest_score = chat_ranking[-1][1]
                assert highest_score != lowest_score
                # Split the chats into better half and worst half
                unique_scores = sorted(set(passed_scores), reverse=True)
                mid_index = len(unique_scores) // 2
                better_scores = unique_scores[:mid_index]
                worse_scores = unique_scores[mid_index:]

                # Collect chats with better scores
                better_half = [
                    (chat, score) for chat, score in chat_ranking if score in better_scores
                ]
                # Collect chats with worse scores
                worse_half = [
                    (chat, score) for chat, score in chat_ranking if score in worse_scores
                ]
                # Pair up better scoring passed chats with worse scoring passed chats
                for (chosen_chat, chosen_score), (rejected_chat, rejected_score) in zip(
                    better_half, worse_half
                ):
                    pairs.append((chosen_chat, rejected_chat, chosen_score))

        if failed_chats and passed_chats:
            # passed_chats_with_scores = list(zip(passed_chats, passed_scores))
            for (chosen_chat, chosen_score), rejected_chat in zip(
                cycle(chat_ranking), failed_chats
            ):
                pairs.append((chosen_chat, rejected_chat, chosen_score))

        for pair in pairs:
            assert len(pair) == 3
            assert type(pair[0]) == list and type(pair[1]) == list
            assert type(pair[0][0]) == dict and type(pair[1][0]) == dict
            assert type(pair[2]) in [int, float, np.int64, np.float64, np.float32]
            # Add pairs to the dataset
            self.add_chat(pair[0], pair[1], pair[2])

        if cfg.wandb:
            wandb.log({"dpodata/num_of_pairs": len(pairs)})

    def add_chat(self, chosen_conv, rejected_conv, chosen_chat_score):
        assert type(chosen_conv) == list and type(rejected_conv) == list
        assert (
            len(chosen_conv) == len(rejected_conv) == 2
            or len(chosen_conv) == len(rejected_conv) == 3
        )
        assert type(chosen_conv[0]) == dict and type(rejected_conv[0]) == dict
        if len(chosen_conv) == 3:
            assert chosen_conv[0]["role"] == "system" and rejected_conv[0]["role"] == "system"
            assert chosen_conv[1]["role"] == "user" and rejected_conv[1]["role"] == "user"
            assert chosen_conv[2]["role"] == "assistant" and rejected_conv[2]["role"] == "assistant"
        else:
            assert chosen_conv[0]["role"] == "user" and rejected_conv[0]["role"] == "user"
            assert chosen_conv[1]["role"] == "assistant" and rejected_conv[1]["role"] == "assistant"
        assert chosen_conv[0]["content"] == rejected_conv[0]["content"]
        assert chosen_conv[1]["content"] == rejected_conv[1]["content"]
        assert chosen_conv[2]["content"] != rejected_conv[2]["content"]
        assert type(chosen_conv[0]["content"]) == str and type(rejected_conv[0]["content"]) == str
        assert type(chosen_conv[1]["content"]) == str and type(rejected_conv[1]["content"]) == str
        assert type(chosen_conv[2]["content"]) == str and type(rejected_conv[2]["content"]) == str
        prompt = chosen_conv[:-1]
        assert prompt == rejected_conv[:-1], "Prompt mismatch between chosen and rejected convs"

        chosen_response = [chosen_conv[-1]]
        rejected_response = [rejected_conv[-1]]

        # Validate that all components are present
        if prompt and chosen_response and rejected_response:
            self.train_dataset["prompt"].append(prompt)
            self.train_dataset["chosen"].append(chosen_response)
            self.train_dataset["rejected"].append(rejected_response)
            self.train_dataset["chosen_chat_score"].append(chosen_chat_score)
        else:
            assert False, "Incomplete DPO datapoint"

    def add_batch_data(self, cfg, passed_function_classes, failed_function_classes):
        all_function_classes = copy.deepcopy(passed_function_classes + failed_function_classes)

        def group_functions_by_prompt(functions):
            # Dictionary to store functions grouped by prompt_num
            prompt_groups = {}

            # Group functions by prompt_num
            for func_class in functions:
                prompt_num = func_class.prompt_num
                if prompt_num not in prompt_groups:
                    prompt_groups[prompt_num] = []
                prompt_groups[prompt_num].append(func_class)

            # Convert groups to sorted list
            grouped_functions = []
            for prompt_num in sorted(prompt_groups.keys()):
                grouped_functions.append(prompt_groups[prompt_num])

            return grouped_functions

        grouped_functions = group_functions_by_prompt(all_function_classes)

        # Sort the functions that have the same prompt_num together in a separate list
        for functions_with_same_prompt in grouped_functions:
            passed_function_classes_same_prompt = []
            failed_function_classes_same_prompt = []
            for function_class in functions_with_same_prompt:
                if function_class.fail_flag == 0:
                    passed_function_classes_same_prompt.append(function_class)
                else:
                    failed_function_classes_same_prompt.append(function_class)
            self.construct_and_add_pairs(
                cfg, passed_function_classes_same_prompt, failed_function_classes_same_prompt
            )

    def get_dataset_above_threshold(self, threshold):
        filtered_dataset = {"prompt": [], "chosen": [], "rejected": [], "chosen_chat_score": []}
        for prompt, chosen, rejected, score in zip(
            self.train_dataset["prompt"],
            self.train_dataset["chosen"],
            self.train_dataset["rejected"],
            self.train_dataset["chosen_chat_score"],
        ):
            if score >= threshold:
                filtered_dataset["prompt"].append(prompt)
                filtered_dataset["chosen"].append(chosen)
                filtered_dataset["rejected"].append(rejected)
                filtered_dataset["chosen_chat_score"].append(score)
        assert (
            len(filtered_dataset["prompt"])
            == len(filtered_dataset["chosen"])
            == len(filtered_dataset["rejected"])
            == len(filtered_dataset["chosen_chat_score"])
        )
        dpo_dataset = Dataset.from_dict(filtered_dataset)
        # Drop the scores since they are not needed for training
        dpo_dataset = dpo_dataset.remove_columns("chosen_chat_score")
        return dpo_dataset

    def get_highest_score(self):
        return max(self.train_dataset["chosen_chat_score"])

    def __len__(self):
        return len(self.train_dataset["prompt"])


