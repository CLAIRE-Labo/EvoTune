# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A programs database that implements the evolutionary algorithm."""
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
from typing import Any

from absl import logging
import numpy as np
import scipy
import wandb

Signature = tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f"`logits` contains non-finite value(s): {non_finites}")
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    # Ensure that probabilities sum to 1 to prevent error in `np.random.choice`.
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1 :])
    return result


class ProgramsDatabase:
    """A collection of programs, organized as islands."""

    def __init__(
        self,
        config,
    ) -> None:
        self._config = config
        self.num_islands = config.num_islands

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(self.num_islands):
            self._islands.append(
                Island(config.functions_per_prompt, config.temp_sampling_flag, config.temp)
            )
            # config.cluster_sampling_temperature_init,
            # config.cluster_sampling_temperature_period))
        self._best_score_per_island = [-float("inf")] * self.num_islands
        self._best_program_per_island = [None] * self.num_islands

        # self._last_reset_time: float = time.time()

    def get_prompt(self, island_id: int | None = None, percentile=1.0):
        """Returns a prompt containing implementations from one chosen island."""
        if island_id == None:
            island_id = np.random.randint(len(self._islands))
        sorted_implementations, scores, temperature, probabilities = self._islands[
            island_id
        ].get_prompt(percentile)
        return sorted_implementations, scores, island_id, temperature, probabilities

    def _register_program_in_island(
        self,
        program,
        island_id: int,
        score,
    ) -> None:
        """Registers `program` in the specified island."""
        self._islands[island_id].register_program(program, score)
        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_score_per_island[island_id] = score
            print(f"Best score of island {island_id} increased to {score}")
            return True
        else:
            return False

    def register_program(
        self,
        program,
        score,
        island_id: int | None,
    ) -> None:
        """Registers `program` in the database."""
        # In an asynchronous implementation we should consider the possibility of
        # registering a program on an island that had been reset after the prompt
        # was generated. Leaving that out here for simplicity.
        if island_id is None:
            # This is a program added at the beginning, so adding it to all islands.
            for island_id in range(len(self._islands)):
                self._register_program_in_island(program, island_id, score)
            return False
        else:
            flag_writetxt = self._register_program_in_island(program, island_id, score)

        # Check whether it is time to reset an island.
        # if (time.time() - self._last_reset_time > self._config.reset_period):
        #   self._last_reset_time = time.time()
        #   self.reset_islands()
        return flag_writetxt

    # def reset_islands(self) -> None:
    #   """Resets the weaker half of islands."""
    #   print("Reseting islands")
    #   # We sort best scores after adding minor noise to break ties.
    #   indices_sorted_by_score: np.ndarray = np.argsort(
    #       self._best_score_per_island +
    #       np.random.randn(len(self._best_score_per_island)) * 1e-6)
    #   num_islands_to_reset = self._config.num_islands // 2
    #   reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
    #   keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
    #   for island_id in reset_islands_ids:
    #     self._islands[island_id] = Island(
    #         #self._template,
    #         #self._function_to_evolve,
    #         self._config.functions_per_prompt,
    #         self._config.cluster_sampling_temperature_init,
    #         self._config.cluster_sampling_temperature_period)
    #     self._best_score_per_island[island_id] = -float('inf')
    #     founder_island_id = np.random.choice(keep_islands_ids)
    #     founder = self._best_program_per_island[founder_island_id]
    #     founder_score = self._best_score_per_island[founder_island_id]
    #     self._register_program_in_island(founder, island_id, founder_score)

    @property
    def get_best_score(self):
        return max(self._best_score_per_island)

    def get_best_score_per_island(self, island_id):
        return self._best_score_per_island[island_id]

    @property
    def total_num_programs(self):
        return sum([island._num_programs for island in self._islands])


class Island:
    """A sub-population of the programs database."""

    def __init__(
        self,
        functions_per_prompt: int,
        temp_sampling_flag: bool,
        temp: float,
        # cluster_sampling_temperature_init: float,
        # cluster_sampling_temperature_period: int,
    ) -> None:
        self._functions_per_prompt: int = functions_per_prompt
        # self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        # self._cluster_sampling_temperature_period = (
        #     cluster_sampling_temperature_period)
        self.temp_sampling_flag = temp_sampling_flag
        self.temp = temp

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0

    def register_program(
        self,
        program,
        score,
    ) -> None:
        """Stores a program on this island, in its appropriate cluster."""
        signature = score
        if signature not in list(self._clusters.keys()):
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)
        self._num_programs += 1

    def get_prompt(self, percentile):
        """Constructs a prompt containing functions from this island."""
        cluster_scores = np.array(list(self._clusters.keys()))

        # At the beginning of an experiment when we have few clusters, place fewer
        # programs into the prompt.
        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        # cfg.temp_sampling_flag determines whether to sample the clusters or take the top scoring clusters
        # Usually, we want to sample the clusters to encourage diversity
        if self.temp_sampling_flag:

            # Convert scores to probabilities using softmax with temperature schedule.
            # period = self._cluster_sampling_temperature_period
            # temperature = self._cluster_sampling_temperature_init * (
            #     1 - (self._num_programs % period) / period)
            # probabilities = _softmax(cluster_scores, temperature)
            
            # do the sampling only after keeping only the top percentile
            percentile_score = np.percentile(cluster_scores, (100-percentile*100), method='nearest')
            number_of_clusters = len(cluster_scores)
            num_of_clusters_in_top_percentile = (cluster_scores > percentile_score).sum()
            if num_of_clusters_in_top_percentile > functions_per_prompt:
                cluster_scores = cluster_scores[cluster_scores > percentile_score]

                # Count how many programs in the clusters are we sampling from
                number_of_programs_in_top_percentile = 0
                for score in cluster_scores:
                    number_of_programs_in_top_percentile += len(self._clusters[score]._programs)

                logging.info(f"Using the top percentile score: {percentile_score}")
                logging.info(f"Number of clusters in the island: {number_of_clusters}")
                logging.info(f"Number of clusters in the top percentile: {num_of_clusters_in_top_percentile}")
                logging.info(f"Number of programs in the top percentile: {number_of_programs_in_top_percentile}")

            probabilities = _softmax(cluster_scores, self.temp)

            # Add a small constant to ensure no zero probabilities
            epsilon = 1e-20
            probabilities += epsilon
            probabilities /= probabilities.sum()
            if len(probabilities) < functions_per_prompt:
                idx = np.random.choice(
                    len(cluster_scores), size=functions_per_prompt, p=probabilities, replace=True
                )
            else:
                idx = np.random.choice(
                    len(cluster_scores), size=functions_per_prompt, p=probabilities, replace=False
                )
            sampled_probabilities = [probabilities[i] for i in idx]
        else:
            sorted_indices = np.argsort(cluster_scores)[::-1]
            # Select the top clusters
            idx = sorted_indices[:functions_per_prompt]
            sampled_probabilities = [-1.0] * functions_per_prompt

        chosen_scores = [cluster_scores[i] for i in idx]

        implementations = []
        scores = []
        for score in chosen_scores:
            cluster = self._clusters[score]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        sorted_scores = [scores[i] for i in indices]
        # version_generated = len(sorted_implementations) + 1

        return sorted_implementations, sorted_scores, self.temp, sampled_probabilities


class Cluster:
    """A cluster of programs on the same island and with the same score."""

    def __init__(self, score, program):
        self._score = score
        self._programs = [program]
        self._lengths = [len(str(program))]

    @property
    def score(self) -> float:
        """Reduced score of the signature that this cluster represents."""
        return self._score

    def register_program(self, program) -> None:
        """Adds `program` to the cluster."""
        # Do not store duplicates.
        if program not in self._programs:
            self._programs.append(program)
            self._lengths.append(len(str(program)))

    def sample_program(self):
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
            max(self._lengths) + 1e-6
        )
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        chosen_idx = np.random.choice(len(self._programs), p=probabilities)
        return self._programs[chosen_idx]
