import heapq
import os
import pickle


def load_all_programsbanks_scores(programbanks_paths, score_threshold=-1000, return_as_islands=False):
    # Create a max-heap by storing negative round numbers
    programbank_heap = []
    for programbank_path in programbanks_paths:
        programbank_round = int(programbank_path.split("_")[-1].split(".")[0])
        heapq.heappush(programbank_heap, (programbank_round, programbank_path))

    programbanks = []
    actual_rounds = []
    while programbank_heap:
        actual_round, programbank_path = heapq.heappop(programbank_heap)
        with open(programbank_path, 'rb') as f:
            programbanks.append(pickle.load(f))
            actual_rounds.append(actual_round)

    programbanks_scores = []
    for programbank in programbanks:
        scores = [] if return_as_islands else [[]]
        for idx, island in enumerate(programbank._islands):
            if not return_as_islands:
                for cluster_score, cluster in island._clusters.items():
                    if cluster_score < score_threshold:
                        continue
                    scores[0].extend([cluster_score] * len(cluster._programs))
            else:
                island_scores = []
                for cluster_score, cluster in island._clusters.items():
                    if cluster_score < score_threshold:
                        continue
                    island_scores.extend([cluster_score] * len(cluster._programs))
                scores.append(island_scores)
        programbanks_scores.append(scores)

    return programbanks_scores, actual_rounds

if __name__ == "__main__":
    directory = "out/tsp/evotune"
    assert os.path.exists(directory)

    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files]

    score_data_per_round_evo, round_nums = load_all_programsbanks_scores(files, score_threshold=-1000,
                                                                         return_as_islands=False
                                                                         )
    print()