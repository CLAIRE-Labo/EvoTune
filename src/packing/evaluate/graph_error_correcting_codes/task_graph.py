from itertools import product
import numpy as np
from typing import Callable
from datetime import datetime
import math
import networkx as nx

def generate_boolean_cube(n):
    """Generates all binary strings of length n."""
    return ["".join(map(str, bits)) for bits in product([0, 1], repeat=n)]


def hamming_distance(s1, s2):
    """Calculates the Hamming distance between two equal-length binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def edges_to_adjacency_list(edges, total_nodes):
    """
    Constructs Graph as a dictionary of the following format-

    graph[VertexNumber V] = list[Neighbors of Vertex V]
    """
    graph = {i: [] for i in total_nodes}

    for i in range(len(edges)):
        v1, v2 = edges[i]

        graph[v1].append(v2)
        graph[v2].append(v1)
    return graph


def construct_graph(vertices, max_distance):
    """ Constructs a graph based on Hamming distance criteria. """
    edges = []
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            # WARNING: Here be careful if < or <= is used!
            if hamming_distance(vertices[i], vertices[j]) < max_distance:
                edges.append((vertices[i], vertices[j]))

    graph = edges_to_adjacency_list(edges,vertices)
    return graph,edges


def MaximalIndependentVertexSet(graph):
    """
    Function for finding the exact maximal independent set in a graph.
    Taken from https://www.geeksforgeeks.org/maximal-independent-set-in-an-undirected-graph/
    """
    # Base Case - Given Graph has no nodes
    if len(graph) == 0:
        return []

    # Base Case - Given Graph has 1 node
    if len(graph) == 1:
        return [list(graph.keys())[0]]

    # Select a vertex from the graph
    vCurrent = list(graph.keys())[0]

    ## Case 1 - Proceed removing the selected vertex from the Maximal Set
    graph2 = dict(graph)

    # Delete current vertex from the Graph
    del graph2[vCurrent]

    # Recursive call - Gets Maximal Set, assuming current Vertex not selected
    res1 = MaximalIndependentVertexSet(graph2)

    ## Case 2 - Proceed considering the selected vertex as part of the Maximal Set

    # Loop through its neighbours
    for v in graph[vCurrent]:

        # Delete neighbor from the current subgraph
        if v in graph2:
            del graph2[v]

    # This result set contains VFirst, and the result of recursive call assuming neighbors of vFirst are not selected
    res2 = [vCurrent] + MaximalIndependentVertexSet(graph2)

    # Our final result is the one which is bigger, return it
    if len(res1) > len(res2):
        return res1
    return res2


# def turan_theorem(vertices, edges):
#     """Applies Turán's theorem to estimate the minimum size of the largest independent set."""
#     V = len(vertices)
#     E = len(edges)
#     return V**2 / (2 * E + V)

# Function to run
# def evaluate_func(vertices: list[str], n: int, d: int, v: int, priority_function: Callable, eval_exact: bool = False):
#     """
#     n: int, Dimension of the Boolean cube
#     d: int, Maximum Hamming distance for graph edges
#     v: int, Number of vertices to choose

#     Returns the the size of the largest independent set. The first value we output is the one we are optimizing for, and the second one is for logging purposes. 
#     """
#     # Choose vertices based on a priority function
#     priorities = [priority_function(vertex) for vertex in vertices]
#     # priorities = priority_function(vertices, n)

#     # Sort vertices based on priority
#     sorted_vertices = [
#         vertex for _, vertex in sorted(zip(priorities, vertices), reverse=True)
#     ]
#     print("Sorted")

#     # Only take the first v vertices
#     chosen_vertices = sorted_vertices[:v]

#     # Construct the graph with the specified maximum Hamming distance
#     # graph = construct_graph(chosen_vertices, d)
#     graph, edges = construct_graph(chosen_vertices, d)
#     print("Graph constructed")

#     # Calculate the size of the largest independent set
#     if len(edges) == 0:
#         print("All selected vertices are already more than d distance apart")
#         exact_max_independent_set_size = len(chosen_vertices)
#         approx_max_independent_set_size = None
    
#     else:
#         approx_max_independent_set_size = turan_theorem(chosen_vertices, edges)

#         if n <= 5 or eval_exact:
#             time_now = datetime.now()
#             maximal_independent_set = MaximalIndependentVertexSet(graph)
#             exact_max_independent_set_size = len(maximal_independent_set)
#             time_end = datetime.now()
#             print(f"Time taken to find the exact maximal independent set: {time_end - time_now}")

#         else:
#             exact_max_independent_set_size = None

#     if exact_max_independent_set_size is not None:
#         k_exact = np.round(math.log2(exact_max_independent_set_size), 7)
#     else: 
#         k_exact = None
    
#     if approx_max_independent_set_size is not None:
#         k_approx = np.round(math.log2(approx_max_independent_set_size), 7)
#     else: 
#         k_approx = None

#     if k_exact is not None:
#         return k_exact, k_approx
#     else:
#         return k_approx, k_exact


# TODO: fix logging into results queue
def evaluate_func(args, vertices: list[str], priority_function: Callable):
    """
    n: int, Dimension of the Boolean cube
    d: int, Maximum Hamming distance for graph edges
    v: int, Number of vertices to choose

    Returns the the size of the largest independent set. The first value we output is the one we are optimizing for, and the second one is for logging purposes. 
    """
    n = args.n
    d = args.d
    v = args.threshold
    # Choose vertices based on a priority function
    #priorities = priority_function(vertices)
    priorities = [priority_function(vertex) for vertex in vertices]
    # priorities = priority_function(vertices, n)

    # Sort vertices based on priority
    sorted_vertices = [
        vertex for _, vertex in sorted(zip(priorities, vertices), reverse=True)
    ]
    print("Sorted")

    # Only take the first v vertices
    chosen_vertices = sorted_vertices[:v]

    # Construct the graph with the specified maximum Hamming distance
    # graph = construct_graph(chosen_vertices, d)
    graph, edges = construct_graph(chosen_vertices, d)
    print("Graph constructed")

    # Calculate the size of the largest independent set
    if len(edges) == 0:
        print("All selected vertices are already more than d distance apart")
        exact_max_independent_set_size = len(chosen_vertices)
    
    else:
        time_now = datetime.now()
        maximal_independent_set = MaximalIndependentVertexSet(graph)
        exact_max_independent_set_size = len(maximal_independent_set)
        time_end = datetime.now()
        print(f"Time taken to find the exact maximal independent set: {time_end - time_now}")


    if exact_max_independent_set_size is not None:
        k_exact = np.round(math.log2(exact_max_independent_set_size), 7)
    else: 
        k_exact = None
    
    return k_exact    


if __name__ == "__main__":

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "identity":0,
        "n": 7,
        "d": 3,
        "keep_perc": 0.75,
    })
    cfg.threshold = int(cfg.keep_perc * 2**cfg.n)


    def priority_function_v2(element: str) -> float:
        """
        Calculates the priority score for a vertex within the Boolean n-dimensional cube based on its binary representation.

        The priority score determines the order of vertex addition to a contact graph. The higher the priority score, 
        the higher the chance the vertex will be added to the graph.

        Parameters:
        - element (str): A binary string representing a vertex in the Boolean n-dimensional cube. Each character in the string is either '0' or '1'.

        Returns:
        - float: The priority score for the given vertex.
        """

        # Check if the input string is valid
        if not set(element).issubset({'0', '1'}):
            raise ValueError("Invalid binary string")

        # Calculate the number of consecutive 0s and 1s
        consecutive_runs = [0]
        for bit in element:
            if bit == consecutive_runs[-1]:
                consecutive_runs[-1] += 1
            else:
                consecutive_runs.append(1)
        consecutive_0s = max(consecutive_runs) if 0 in consecutive_runs else 0
        consecutive_1s = max(consecutive_runs) if 1 in consecutive_runs else 0

        # Calculate the number of unique bits
        unique_bits = len(set(element))

        # Calculate the balance of 0s and 1s
        num_zeros = element.count('0')
        num_ones = len(element) - num_zeros
        if num_zeros == 0 or num_ones == 0:
            balance = 0
        else:
            balance = 1 - np.abs(num_zeros - num_ones) / len(element)

        # Calculate the location of 0s and 1s
        zero_indices = np.array([i for i, bit in enumerate(element) if bit == '0'])
        one_indices = np.array([i for i, bit in enumerate(element) if bit == '1'])
        if zero_indices.size > 0 and one_indices.size > 0:
            zero_center = np.mean(zero_indices)
            one_center = np.mean(one_indices)
            center_score = min(zero_center, one_center)
        elif zero_indices.size > 0:
            center_score = len(element)
        elif one_indices.size > 0:
            center_score = len(element)
        else:
            center_score = len(element)

        # Calculate the priority score
        priority = round(0.3 * consecutive_0s + 0.2 * consecutive_1s + 0.2 * unique_bits + 0.1 * balance + 0.1 * center_score, 5)

        return priority


    #########
    # Insert a priority function with experiment parameters below here
    #########
    # n = 10
    # d = 2
    # keep_perc = 0.5
    # def priority_function_v2(element: str) -> float:
    #     """
    #     Calculates the priority score for a vertex within the Boolean n-dimensional cube based on its binary representation.

    #     The priority score is based on the Hamming weight, its evenness, and the number of adjacent vertices.

    #     Parameters:
    #     - element (str): A binary string representing a vertex in the Boolean n-dimensional cube. Each character in the string is either '0' or '1'.

    #     Returns:
    #     - float: The priority score for the given vertex.
    #     """
        
    #     # Calculate the Hamming weight (number of 1s) in the binary string
    #     hamming_weight = sum(c == '1' for c in element)
        
    #     # Calculate the number of adjacent vertices
    #     adjacent_vertices = sum(element[i] != element[j] for i in range(len(element)) for j in range(len(element)) if i != j)
        
    #     # Calculate the evenness of the Hamming weight
    #     evenness = 1 if hamming_weight % 2 == 0 else 0
        
    #     # Calculate the priority score based on the Hamming weight, its evenness, and the number of adjacent vertices
    #     if hamming_weight == 0:
    #         priority_score = 1
    #     else:
    #         priority_score = (1 / (1 + abs((len(element) / 2) - hamming_weight))) * (1 / (1 + abs((len(element) * (len(element) - 1)) / 2 - adjacent_vertices))) * evenness
        
    #     return priority_score

    #########
    # End of priority function
    #########


    vertices = generate_boolean_cube(cfg.n)

    k_exact = evaluate_func(cfg, vertices, priority_function_v2)
    print("\n")
    print("Evaluation of the priority_function_v2")
    print("k exact = ", k_exact)
        