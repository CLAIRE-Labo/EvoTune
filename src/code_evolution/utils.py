import heapq
import pickle
import re


def get_evenly_spaced_function_from_programbank(programbanks_paths, score_threshold=-14):
    # Create a max-heap by storing negative round numbers
    programbank_heap = []
    for programbank_path in programbanks_paths:
        programbank_round = int(programbank_path.split("_")[-1].split(".")[0])
        heapq.heappush(programbank_heap, (programbank_round, programbank_path))

    programbanks = []
    programbanks_rounds = [heap_val[0] for heap_val in programbank_heap]
    while programbank_heap:
        _, programbank_path = heapq.heappop(programbank_heap)
        with open(programbank_path, 'rb') as f:
            programbanks.append(pickle.load(f))

    # Find the island with the highest scoring cluster in the last programbank
    best_score = float('-inf')
    best_island_idx = None

    for island_idx, island in enumerate(programbanks[-1]._islands):
        for cluster_score in island._clusters.keys():
            if cluster_score > best_score:
                best_score = cluster_score
                best_island_idx = island_idx

    # all_clusters = set()
    selected_programs = []
    selected_program_scores = []
    selected_program_rounds = []
    for i, programbank in enumerate(programbanks):
        # Get that best island
        best_island = programbank._islands[best_island_idx]

        # Sort clusters by score ascending
        cluster_scores_sorted = [score for score in sorted(best_island._clusters.keys(), reverse=True) if
                                 score > score_threshold] # and score not in all_clusters]
        # all_clusters.update(cluster_scores_sorted)

        if len(cluster_scores_sorted) == 0:
            print(f'Found no new-scoring programs for round {programbanks_rounds[i]}, continuing.')
            continue

        # We pick the best-scoring function from this island at this iteration
        top_score_cluster = cluster_scores_sorted[0]
        program = best_island._clusters[top_score_cluster]._programs[-1]  # Get the first program

        selected_programs.append(program)
        selected_program_scores.append(top_score_cluster)
        selected_program_rounds.append(programbanks_rounds[i])

    return selected_programs, selected_program_scores, programbanks_rounds


def shorten_function_code(code_str: str, cutoff_lines: int = 30, exclude_last: int = 0) -> str:
    # Find the function definition by locating 'def priority' and capturing until the colon
    function_match = re.search(r"def priority[\s\S]*?:", code_str)
    if not function_match:
        raise ValueError("Function definition not found.")

    # Start the shortened code with the cleaned function signature
    shortened_code = "def priority(...):\n\t"

    # Extract everything after the function definition
    function_body_start = function_match.end()
    function_body = code_str[function_body_start:].strip()

    # Remove all docstrings or triple-quoted blocks inside the function body
    function_body = re.sub(r'"""[\s\S]*?"""', '', function_body)

    # Split into lines for processing
    body_lines = function_body.splitlines()

    # Remove any stray annotation lines right after the def line
    # For JSSP:
    while body_lines and (body_lines[0].strip().endswith(",")
                          or body_lines[0].strip().startswith("blocks")
                          or body_lines[0].strip().startswith("action_mask")
                          or body_lines[0].strip().startswith(") -> np.ndarray")):
        body_lines.pop(0)
    # For FlatPack:
    # while body_lines and (
    #         body_lines[0].strip().startswith("np.") or body_lines[0].strip().endswith(") -> np.ndarray:")):
    #     body_lines.pop(0)

    # Exclude the last 'exclude_last' lines from consideration
    if exclude_last > 0:
        body_lines = body_lines[:-exclude_last]

    # Adjust cutoff to consider that lines were excluded
    adjusted_cutoff = cutoff_lines

    # If the body is too long, keep only the last 'adjusted_cutoff' lines and prepend '...'
    if len(body_lines) > adjusted_cutoff:
        kept_lines = ["...\n"] + body_lines[-adjusted_cutoff:]
    else:
        kept_lines = body_lines

    # Reconstruct the shortened body
    shortened_body = "\n".join(kept_lines)
    shortened_code += shortened_body

    return shortened_code


if __name__ == "__main__":
    code_versions, code_scores, round_nums = get_evenly_spaced_function_from_programbank(
        ["out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_100.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_500.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_1000.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_1500.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_1600.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2000.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2300.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2600.pkl",
         "out/flatpack/evotune/programbank_llama32arxivs2_4_tsp_2700.pkl",
         ], score_threshold=-20
    )
    print()
    code_versions = [shorten_function_code(v, cutoff_lines=45, exclude_last=0) for v in code_versions]
    print()