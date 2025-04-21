import random
import json
from typing import List, Dict
import copy
from datasets import Dataset
import numpy as np

class ChatDataBuffer:
    def __init__(self):
        self.data = []
        self.scores = []
        self.scores_since_finetune = []

    def add_batch_data(self, passed_function_classes, only_correct):
        # Add chats to the SFT buffer
        for passed_class in passed_function_classes:
            # # Only add correct functions to the SFT buffer
            if only_correct:
                if passed_class.correct_flag == 1:
                    self.add_data(passed_class.chat, passed_class.score)
            else: 
                self.add_data(passed_class.chat, passed_class.score)


    def add_data(self, chat: Dict[str, str], score: float):
        """
        Add a single chat instance to the buffer.
        Each chat is expected to be a list of messages, where each message is a dictionary with keys "role" and "content".
        """
        assert isinstance(chat, list)
        assert len(chat) in [2, 3]
        assert type(score) in [int, float, np.int64, np.float64, np.float32]
        if len(chat) == 2:
            assert chat[0]["role"] == "user"
            assert chat[1]["role"] == "assistant"
            assert type(chat[0]["content"]) == str
            assert type(chat[1]["content"]) == str
        else:
            assert chat[0]["role"] == "system"
            assert chat[1]["role"] == "user"
            assert chat[2]["role"] == "assistant"
            assert type(chat[0]["content"]) == str
            assert type(chat[1]["content"]) == str
            assert type(chat[2]["content"]) == str
        self.data.append(chat)
        self.scores.append(score)
        self.scores_since_finetune.append(score)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.scores[index]

    def get_highest_score(self):
        """
        Find and return the highest score in the buffer.
        """
        if not self.scores:
            return None
        return max(self.scores)

    # Filter out messages above a threshold score
    def get_dataset_above_threshold(self, threshold):
        assert len(self.data) == len(self.scores)
        chats = copy.deepcopy(self.data)
        scores = copy.deepcopy(self.scores)
        assert len(chats) == len(scores)

        funetuning_messages = [
            chats[i]
            for i, score in enumerate(scores)
            if score >= threshold
        ]
        data_dict_chats = [{"messages": row} for row in funetuning_messages]
        train_dataset = Dataset.from_list(data_dict_chats)
        return train_dataset

