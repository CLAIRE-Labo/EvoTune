from itertools import product
import itertools
import numpy as np
from typing import Callable
from datetime import datetime
import math
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set
from networkx.algorithms import maximal_independent_set as find_max_independent_set
from networkx.algorithms.approximation import maximum_independent_set as find_approx_max_independent_set
import numpy as np
from scipy.spatial import KDTree
#import rustworkx as rx
import numpy as np
from scipy.spatial.distance import pdist, squareform
import networkx as nx
from sklearn.neighbors import BallTree
import random


def generate_boolean_cube(n):
    """Generates all binary strings of length n."""
    return ["".join(map(str, bits)) for bits in product([0, 1], repeat=n)]


def generate_hamming_cube(dimension):
    """
    Generate a list of lists with Hamming cube coordinates of the given dimension.

    Parameters:
    dimension (int): The dimension of the Hamming cube.

    Returns:
    List[List[int]]: A list of lists containing the coordinates of the Hamming cube.
    """
    # Use itertools.product to generate all possible binary combinations of the given dimension
    hamming_cube = list(itertools.product([0, 1], repeat=dimension))
    
    # Convert each tuple to a list
    hamming_cube = [list(coord) for coord in hamming_cube]
    
    return hamming_cube


def hamming_distance(s1, s2):
    """Calculates the Hamming distance between two equal-length binary strings."""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


# def edges_to_adjacency_list(edges):
#     """
#     Constructs Graph as a dictionary of the following format-

#     graph[VertexNumber V] = list[Neighbors of Vertex V]
#     """
#     graph = dict([])

#     for i in range(len(edges)):
#         v1, v2 = edges[i]

#         if v1 not in graph:
#             graph[v1] = []
#         if v2 not in graph:
#             graph[v2] = []

#         graph[v1].append(v2)
#         graph[v2].append(v1)
#     return graph

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


def turan_theorem(vertices, edges):
    """Applies Turán's theorem to estimate the minimum size of the largest independent set."""
    V = len(vertices)
    E = len(edges)
    return V**2 / (2 * E + V)


def transform_binary_strings_to_lists(binary_strings):
    """
    Transform a list of binary strings into a list of lists of integers.

    Parameters:
    binary_strings (List[str]): A list of binary strings.

    Returns:
    List[List[int]]: A list of lists containing integers.
    """
    # Use list comprehension to transform each string to a list of integers
    transformed_list = [[int(bit) for bit in binary_string] for binary_string in binary_strings]
    
    return transformed_list

def create_hamming_graph(points, threshold):
    """
    Create a graph where an edge is added between two points if their Hamming distance is less than the given threshold.
    
    Parameters:
    points (list of list of int): A list of points on the Hamming cube.
    threshold (float): The maximum Hamming distance to allow an edge.
    
    Returns:
    networkx.Graph: A graph with points as nodes and edges between nodes with Hamming distance less than threshold.
    """
    # Convert points to numpy array for efficient computation
    points_array = np.array(points)
    
    # Compute pairwise Hamming distances
    hamming_distances = pdist(points_array, metric='hamming')
    
    # Convert to a square form distance matrix
    distance_matrix = squareform(hamming_distances)
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(len(points)):
        G.add_node(i, point=points[i])
    
    # Add edges based on the distance threshold
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if distance_matrix[i, j] < threshold:
                G.add_edge(i, j)
    
    return G

def create_hamming_graph_with_balltree(points, threshold,n):
    """
    Create a graph where an edge is added between two points if their Hamming distance is less than the given threshold,
    using BallTree for efficient neighbor queries.
    
    Parameters:
    points (list of list of int): A list of points on the Hamming cube.
    threshold (float): The maximum Hamming distance to allow an edge.
    
    Returns:
    networkx.Graph: A graph with points as nodes and edges between nodes with Hamming distance less than threshold.
    """
    # Convert points to numpy array for efficient computation
    points_array = np.array(points)
    
    # Create BallTree with Hamming metric
    tree = BallTree(points_array, metric='hamming')
    
    # Create a graph
    G = nx.Graph()
    
    # Add nodes to the graph
    for i in range(len(points)):
        G.add_node(i, point=points[i])
    
    # Find neighbors within the threshold distance for each point
    #from decimal import Decimal  
    #r = Decimal((threshold)/n)
    r = threshold/n
    epsilon = np.finfo(float).eps
    print("r is", r)

    for i in range(len(points)):
        indices = tree.query_radius([points_array[i]], r=r-epsilon)[0]
        for j in indices:
            if i < j:  # Ensure each edge is added only once
                G.add_edge(i, j)
    
    return G

# Function to run
def evaluate_func(vertices: list[str], n: int, d: int, v: int, priority_function: Callable, eval_exact: bool = False):
    """
    n: int, Dimension of the Boolean cube
    d: int, Maximum Hamming distance for graph edges
    v: int, Number of vertices to choose

    Returns the the size of the largest independent set. The first value we output is the one we are optimizing for, and the second one is for logging purposes. 
    """
    # Choose vertices based on a priority function
    priorities = [priority_function(vertex) for vertex in vertices]
    # priorities = priority_function(vertices)

    # Sort vertices based on priority
    sorted_vertices = [
        vertex for _, vertex in sorted(zip(priorities, vertices), reverse=True)
    ]
    print("Sorted")

    # Only take the first v vertices
    chosen_vertices = sorted_vertices[:v]

    chosen_vertices_list = transform_binary_strings_to_lists(chosen_vertices)

    f = len(chosen_vertices_list[0])  # Length of item vector that will be indexed

    # time_now = datetime.now()
    # t = AnnoyIndex(f, 'hamming')
    # # for i in range(1000):
    # #     v = [random.gauss(0, 1) for z in range(f)]
    # for i, v in enumerate(chosen_vertices_list):
    #     t.add_item(i, v)

    # t.build(10) # 10 trees
    #t.save('test.ann')

    # def build_graph_from_annoy(tree, num_elements, distance_threshold):
    #     """
    #     Constructs a graph where nodes are connected if their distance is less than the specified threshold.
        
    #     Parameters:
    #     tree (AnnoyIndex): The Annoy tree.
    #     num_elements (int): The number of elements in the tree.
    #     distance_threshold (float): The distance threshold for connecting nodes.
        
    #     Returns:
    #     nx.Graph: The constructed graph.
    #     """
    #     # Create an empty graph
    #     graph = nx.Graph()
        
    #     # Add all elements as nodes in the graph
    #     graph.add_nodes_from(range(num_elements))
        
    #     # Iterate through each element and find its neighbors within the distance threshold
    #     for i in range(num_elements):
    #         neighbors = tree.get_nns_by_item(i, num_elements, include_distances=True)
    #         for neighbor, distance in zip(*neighbors):
    #             if neighbor > i and distance < distance_threshold:
    #                 graph.add_edge(i, neighbor, weight=distance)
        
    #     return graph


    # neighbors_idx, neighbours_distances = t.get_nns_by_vector(chosen_vertices_list[0],10,include_distances=True)
    # neighbours = [chosen_vertices_list[idx] for idx in neighbors_idx]

    #graph = build_graph_from_annoy(t, len(chosen_vertices_list), d)

    # time_end = datetime.now()
    # print(f"Time taken to construct the graph with ANNOY: {time_end - time_now}")
    

    #tree = KDTree(chosen_vertices)

    #create_hamming_graph
    time_now = datetime.now()
    G_ball  = create_hamming_graph_with_balltree(chosen_vertices_list, d,n)
    print("Edges found in G_ball: ", len(G_ball.edges))
    time_end = datetime.now()
    print(f"Time taken to construct the graph with BallTree: {time_end - time_now}")

    # # RUSTWORKX
    # graph = rx.PyGraph()

    # node_indices = graph.add_nodes_from(chosen_vertices)

    # time_now = datetime.now()
    # for i in range(len(chosen_vertices)):
    #     for j in range(i + 1, len(chosen_vertices)):  # Start from i + 1 to avoid duplicates
    #         if hamming_distance(chosen_vertices[i], chosen_vertices[j]) < d:
    #             graph.add_edge(node_indices[i], node_indices[j], None)
    # time_end = datetime.now()
    # print(f"Time taken to construct the graph: {time_end - time_now}")
    # print("Number of edges: ", graph.num_edges())


    # Create a graph
    # NetworkX graph
    # G = nx.Graph()
    G = nx.Graph()

    # Add nodes to the graph
    G.add_nodes_from(chosen_vertices)

    # Add edges based on the criteria
    time_now = datetime.now()
    for u in G.nodes():
        for v in G.nodes():
            if u < v and hamming_distance(u, v)< d:
                G.add_edge(u, v)

    time_end = datetime.now()
    print(f"Time taken to construct the graph: {time_end - time_now}")
    print("Number of edges: ", G.number_of_edges())
    print("Graph constructed")
    # Add nodes to the graph
    G.add_nodes_from(chosen_vertices)

    # Add edges based on the criteria
    time_now = datetime.now()
    for u in G.nodes():
        for v in G.nodes():
            if u < v and hamming_distance(u, v)< d:
                G.add_edge(u, v)

    time_end = datetime.now()
    print(f"Time taken to construct the graph: {time_end - time_now}")
    print("Number of edges: ", G.number_of_edges())
    print("Graph constructed")

    time_now = datetime.now()
    max_independent_set = find_max_independent_set(G)
    time_end = datetime.now()
    print("Maximal independent set size: ", len(max_independent_set))
    print(f"Time taken to find the random max independent set: {time_end - time_now}")

    # time_now = datetime.now()
    # approx_max_independent_set = find_approx_max_independent_set(G)
    # time_end = datetime.now()
    # print("Maximal independent set size: ", len(approx_max_independent_set))
    # print(f"Time taken to find the approx max independent set: {time_end - time_now}")

    print("Switch to my own implementation\n")

    # Construct the graph with the specified maximum Hamming distance
    # graph = construct_graph(chosen_vertices, d)
    time_now = datetime.now()
    graph, edges = construct_graph(chosen_vertices, d)
    time_end = datetime.now()
    print(f"Time taken to construct the graph: {time_end - time_now}")
    print("Number of edges: ", len(edges))
    print("Graph constructed")

    # Calculate the size of the largest independent set
    if len(edges) == 0:
        print("All selected vertices are already more than d distance apart")
        exact_max_independent_set_size = len(chosen_vertices)
        approx_max_independent_set_size = None
    
    else:
        approx_max_independent_set_size = turan_theorem(chosen_vertices, edges)
        print("Approximate maximal independent set size: ", approx_max_independent_set_size)

        if n <= 5 or eval_exact:
            print(bool(n <= 5))
            print(eval_exact)
            time_now = datetime.now()
            maximal_independent_set = MaximalIndependentVertexSet(graph)
            exact_max_independent_set_size = len(maximal_independent_set)
            time_end = datetime.now()
            print(f"Time taken to find the exact maximal independent set: {time_end - time_now}")
            print("Exact maximal independent set size: ", exact_max_independent_set_size)
            print("Maximal independent set: ", maximal_independent_set)

        else:
            exact_max_independent_set_size = None

    if exact_max_independent_set_size is not None:
        k_exact = np.round(math.log2(exact_max_independent_set_size), 7)
    else: 
        k_exact = None
    
    if approx_max_independent_set_size is not None:
        k_approx = np.round(math.log2(approx_max_independent_set_size), 7)
    else: 
        k_approx = None

    if k_exact is not None:
        return k_exact, k_approx
    else:
        return k_approx, k_exact


    # In the dictionary graph, for every key count the len of the value list
    # dictionary = graph 
    # for key in dictionary:
    #     dictionary[key] = len(dictionary[key])
    # previous_max = 0
    # for key in dictionary:
    #     if dictionary[key] > previous_max:
    #         previous_max = dictionary[key]
    # print("Max degree  is", previous_max) 


if __name__ == "__main__":

    #########
    # Insert a priority function with experiment parameters below here
    #########
    # n = 3 #14
    # d = 2
    # keep_perc = 0.5
    # def priority_function_v2(element: str) -> float:
    #     """
    #     Calculates the priority score for a vertex within the Boolean n-dimensional cube based on its binary representation.

    #     The priority score determines the order of vertex addition to a contact graph. The higher the priority score, the higher the chance the vertex will be added to the graph.

    #     Parameters:
    #     - element (str): A binary string representing a vertex in the Boolean n-dimensional cube. Each character in the string is either '0' or '1'.

    #     Returns:
    #     - float: The priority score for the given vertex.
    #     """
    #     # Calculate the frequency-based priority
    #     one_frequency = sum(map(int, element)).bit_length()
    #     zero_frequency = len(element) - one_frequency
    #     frequency_priority = (one_frequency + one_frequency.bit_length()) / (one_frequency + zero_frequency)  # revised

    #     # Calculate the density-based priority
    #     density = sum(map(int, element)) / len(element)
    #     density_priority = density * (1 - density)  # revised

    #     # Adjust the priority score based on the total number of 1's and 0's and the number of consecutive 1's and 0's
    #     total_consecutive_ones = 0
    #     total_consecutive_zeros = 0
    #     for bit in element:
    #         if bit == '1':
    #             if total_consecutive_ones > 0:
    #                 total_consecutive_ones += 1
    #             else:
    #                 total_consecutive_ones = 1
    #         else:
    #             if total_consecutive_zeros > 0:
    #                 total_consecutive_zeros += 1
    #             else:
    #                 total_consecutive_zeros = 1
    #     adjustment_priority = max(total_consecutive_ones, total_consecutive_zeros) / len(element)  # revised

    #     # Ensure the priority score is in the range [0, 1]
    #     priority_score = frequency_priority + density_priority + adjustment_priority

    #     return priority_score

    n = 14
    d = 3
    keep_perc = 0.5
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
    # End of priority function
    #########


    threshold = int(keep_perc * 2**n)

    vertices = generate_boolean_cube(n)

    #vertices_kdt = generate_hamming_cube(n)
    
    k_exact, k_approx = evaluate_func(vertices, n, d, threshold, priority_function_v2, eval_exact=False)
    print("\n")
    print("Evaluation of the priority_function_v2")
    print("k exact = ", k_exact)
    print("k approx = ", k_approx)
        