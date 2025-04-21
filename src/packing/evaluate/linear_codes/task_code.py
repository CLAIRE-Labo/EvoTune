# # Commented to be able to run on the Apple cluster

# import numpy as np
# from packing.utils.seeding import seed_everything
# from packing.logging.function_class import FunctionClass
# from argparse import Namespace
# import time as time
# # from packing.evaluate.error_correcting_codes.custom_init import custom_init
# from typing import Type
# import logging
# import copy
# import random
# # from sage.all import * ##TODO: turn back on, needed to debug on apple
# import traceback


# #from sage.coding.codecan.autgroup_can_label import LinearCodeAutGroupCanLabel
# # from sage.all import random_matrix, matrix, LinearCode, set_random_seed, GF, gap
# #from packing.evaluate.execute_function import execute_function

# def get_initial_func(cfg):
#     def update_matrix(M):
#         """
#         Updates the input binary matrix M and returns a new binary matrix M_new of the same shape as input.

#         Args:
#             M (numpy.ndarray): A 2D numpy array where each element is either 0 or 1.
#                             This represents the input binary matrix to be updated.

#         Returns:
#             M_new (numpy.ndarray): A 2D numpy array of the same shape as M, where each element is either 0 or 1.
#                         This represents the updated binary matrix. Currently, it is identical to the input matrix M.

#         Note:
#             This is a placeholder function that currently does not modify the input matrix.
#             In a real implementation, this function would contain logic to update the matrix based on specific rules or conditions.
#             Replace the placeholder logic with the actual matrix update logic as required.
#         """
#         M_new = M
#         return M_new
    
#     if cfg.self_ortho:

#         def update_matrix(M):
#             """
#             Updates the input self-orthogonal binary matrix M and returns a new binary 
#             self-orthogonal matrix M_new.

#             The function constructs an orthogonal multiplier matrix over GF(2) and multiplies
#             it with the input matrix M to produce a new self-orthogonal matrix M_new.

#             Args:
#                 M (numpy.ndarray): A self-orthogonal binary matrix of shape (k, n).

#             Returns:
#                 numpy.ndarray: A new self-orthogonal binary matrix of shape (k, n).

#             Raises:
#                 AssertionError: If the resulting matrix is not self-orthogonal.

#             Notes:
#                 - A matrix M_new is self-orthogonal if (M_new @ M_new.T) % 2 == 0.
#                 - The multiplier matrix is constructed to be orthogonal over GF(2), i.e.,
#                 it satisfies (C @ C.T) % 2 == I, where I is the identity matrix.
#                 - Multiplying M by an orthogonal matrix preserves the self-orthogonality of M.
#             """
#             k, n = M.shape

#             # Select four unique random column indices
#             selected_indices = random.sample(range(n), 4)
            
#             # Initialize a binary multiplier matrix with zeros
#             C = np.zeros((n, n), dtype=int)
            
#             # Set ones in the selected positions
#             for i in selected_indices:
#                 for j in selected_indices:
#                     C[i, j] = 1
            
#             # Add the identity matrix modulo 2 to create an orthogonal matrix
#             C = (C + np.identity(n, dtype=int)) % 2

#             # Multiply the multiplier matrix with the input matrix and apply modulo 2
#             M_new = (M @ C) % 2

#             # Verify that the new matrix is self-orthogonal
#             orthogonality_check = (M_new @ M_new.T) % 2
#             assert np.all(orthogonality_check == 0), "The new matrix is not self-orthogonal"

#             return M_new


#     elif cfg.random_init == 1:
#         print("Init function incentivized for randomness")

#         def update_matrix(M):
#             """
#             Perturbs the input binary matrix M my multiplying it with a random invertible matrix.

#             This function generates a random invertible binary matrix U over GF(2) by performing random elementary
#             row operations starting from the identity matrix. It then multiplies U with M over GF(2) to produce
#             the updated matrix M_new.

#             Args:
#                 M (numpy.ndarray): A 2D numpy array with elements 0 or 1, representing the input binary matrix.

#             Returns:
#                 numpy.ndarray: A 2D numpy array of the same shape as M, containing the updated binary matrix. Each element is either 0 or 1.
#             """
#             n = M.shape[0]
#             # Initialize U as the identity matrix
#             U = np.eye(n, dtype=int)
#             # Perform random elementary row operations over GF(2)
#             for _ in range(n * n):
#                 i = np.random.randint(0, n)
#                 j = np.random.randint(0, n)
#                 if i != j:
#                     # Add row j to row i modulo 2
#                     U[i] = (U[i] + U[j]) % 2
#             # Multiply U with M over GF(2)
#             M_new = (U @ M) % 2
#             return M_new

#         # if cfg.unimodular:  # Use unimodular matrix

#         #     def update_matrix(M):
#         #         """
#         #         Perturbs the input binary matrix M my multiplying it with a random invertible matrix.

#         #         Args:
#         #             M (numpy.ndarray): A 2D numpy array where each element is either 0 or 1.
#         #                             This represents the input binary matrix to be updated.

#         #         Returns:
#         #             M_new (numpy.ndarray): A 2D numpy array of the same shape as M, where each element is either 0 or 1.
#         #                         This represents the updated binary matrix.

#         #         """

#         #         M_new = np.matmul(random_matrix(GF(2), M.shape[0], M.shape[0], algorithm="unimodular").numpy(), M)

#         #         return M_new

#     initial_function = update_matrix
#     function_str_to_extract = "update_matrix"

#     # For debugging: check the source code of a function
#     # inspect.getsource(dummy_function)
#     return initial_function, function_str_to_extract


# def generate_input(cfg):
#     set_random_seed(cfg.seed)
#     # if cfg.custom_init:
#     #     input_struct = custom_init(cfg)
#     if cfg.self_ortho:
#         assert (cfg.n % cfg.k) == 0
#         repeats_ones = int(cfg.n / cfg.k)
#         G0 = np.zeros((cfg.k, cfg.n), dtype=int)
#         for j in range(cfg.k):
#             for r in range(repeats_ones):
#                 G0[j, j * repeats_ones + r] = 1
#         input_struct = np.array(G0)
#         # G0 = np.zeros((cfg.k, cfg.n), dtype=int)
#         # for j in range(cfg.k):
#         #     G0[j, 2 * j] = 1
#         #     G0[j, 2 * j + 1] = 1
#         # input_struct_debug = np.array(G0)
#     else:
#         if cfg.identity:
#             shape0 = cfg.k
#             shape1 = cfg.n - cfg.k
#         else:
#             shape0 = cfg.k
#             shape1 = cfg.n

#         if cfg.start_with_zero:
#             G = np.zeros((shape0, shape1))
#             #G = matrix(GF(2), G)

#         elif cfg.start_with_one:
#             assert cfg.start_with_zero == False
#             G = np.ones((shape0, shape1))
#             #G = matrix(GF(2), G)
#         else:
#             #raise ValueError("Gap debugging")
#             # TODO: check how this matrix looks like with Identity
#             G = random_matrix(GF(2), shape0, shape1)
#             input_struct = np.array(G)

#     input_struct = np.array(input_struct, dtype=int)
#     return input_struct


# def orthogonality_check(M):
#     """
#     Checks if the input matrix M is self-orthogonal.

#     Args:
#         M (numpy.ndarray): A binary matrix of shape (k, n).

#     Returns:
#         bool: True if the input matrix is self-orthogonal, False otherwise.
#     """
#     # Compute the matrix product modulo 2
#     orthogonality_check = np.matmul(M, M.T) % 2
#     return np.all(orthogonality_check == 0)

# def is_degenerate(matrix):
#     # Check if two rows are the same
#     rows = [tuple(row) for row in matrix]
#     if len(rows) != len(set(rows)):
#         return True
#     else:
#         return False

# # def construct_cyclic_generator_matrix(cfg, row):
# #     assert isinstance(row, list)
# #     # Correct initial row based on the generator polynomial
# #     row = row + [0] * (cfg.k - 1)  # 11

# #     # Create the generator matrix
# #     G = np.zeros((cfg.k, cfg.n), dtype=int)

# #     for i in range(cfg.k):
# #         G[i] = np.roll(row, -i)
# #     return G


# #     imports = "from numpy.random import permutation"

# #     perturb_func_str = '''
# # def perturb_func(input_struct):
# #     return permutation([1, 4, 9, 12, 15])
# #     '''

# #     globals_dict = {}

# #     # Execute the imports into the globals dictionary
# #     exec(imports, globals_dict)

# #     # Execute the perturbation function string using the same globals dictionary
# #     local_dict = {}
# #     exec(perturb_func_str, globals_dict, local_dict)

# #     # Extract the function from the local dictionary
# #     perturb_func = local_dict.get("perturb_func")

# #     input_struct = [1, 2, 3, 4, 5]
# #     # Import numpy and pass it as an argument
# #     output_struct = perturb_func(input_struct)

# def execute_llm_function(cfg, function_str, imports_str, general_imports_str=""):
#     return None

# GENERAL_IMPORTS = '''
# import random
# import numpy
# import numpy as np
# from itertools import product
# import math
# import scipy
# import scipy.stats
# import scipy.special
# import copy
# '''

# def evaluate_func(cfg, input_struct, function_class: Type[FunctionClass]):
#     #import sage.all
#     #from sage.coding.codecan.autgroup_can_label import LinearCodeAutGroupCanLabel
#     # gap = sage.interfaces.gap.gap
#     #logging.info(f"TASK {function_class.eval.idx_process} running gap")
    

#     ### START
#     print(f"Task id {function_class.eval.idx_process}: Starting evaluation")

#     logging.info(f"TASK {function_class.eval.idx_process} executing imports")

#     perturb_func_str = function_class.function_str
#     imports = function_class.imports_str
                
#     # Execute imports and the function
#     try:
#         # Create a shared globals dictionary
#         globals_dict = {}

#         # Execute general imports
#         exec(GENERAL_IMPORTS, globals_dict)

#         # Execute the imports into the globals dictionary
#         exec(imports, globals_dict)

#         # Execute the perturbation function string using the same globals dictionary
#         local_dict = {}
#         exec(perturb_func_str, globals_dict, local_dict)

#         # Extract the function from the local dictionary
#         perturb_func = local_dict.get(cfg.function_str_to_extract)

#         # exec(imports)
#         # local_dict = {}
#         # exec(perturb_func_str, globals(), local_dict)
#         # perturb_func = local_dict.get(cfg.function_str_to_extract)
#     except Exception as e:
#         tb_str = traceback.format_exc()
#         function_class.fail_flag = 1
#         function_class.fail.reason_imports = 1
#         function_class.score = cfg.failed_score
#         function_class.true_score = cfg.failed_score
#         function_class.fail.exception = tb_str
#         logging.info(f"TASK {function_class.eval.idx_process} failed imports")
#         return function_class

#     # Asserts before applying the perturbation function
#     assert isinstance(input_struct, np.ndarray), f"OUTPUT_STRUCT ERROR: The input structure is not a numpy array: {input_struct}"
#     #function_class.eval.input_struct = input_struct

#     logging.info(f"TASK {function_class.eval.idx_process} applying perturbation function")
#     # Apply the perturbation function
#     try:
#         output_struct = perturb_func(input_struct)
#     except Exception as e:
#         tb_str = traceback.format_exc()
#         function_class.fail_flag = 1
#         function_class.fail.reason_exception = 1
#         function_class.score = cfg.failed_score
#         function_class.true_score = cfg.failed_score
#         function_class.fail.exception = tb_str
#         logging.info(f"TASK {function_class.eval.idx_process} failed perturbation function")
#         return function_class
    
#     logging.info(f"TASK {function_class.eval.idx_process} finished perturbation function")

#     logging.info(f"TASK {function_class.eval.idx_process} type check")
#     # Type check
#     if type(output_struct) != type(input_struct):
#         function_class.fail_flag = 1
#         function_class.fail.reason_type = 1
#         function_class.score = cfg.failed_score
#         function_class.true_score = cfg.failed_score
#         return function_class

#     logging.info(f"TASK {function_class.eval.idx_process} shape check")
#     # Shape check
#     if input_struct.shape != output_struct.shape:
#         function_class.fail_flag = 1
#         function_class.score = cfg.failed_score
#         function_class.true_score = cfg.failed_score
#         function_class.fail.reason_shape = 1
#         return function_class
#     output_shape0, output_shape1 = output_struct.shape
#     function_class.eval.shape0 = output_shape0
#     function_class.eval.shape1 = output_shape1

#     logging.info(f"TASK {function_class.eval.idx_process} binarizing output matrix")
#     # Convert elements of output_struct to 0 or 1
#     def binarize_matrix(output_matrix):
#         output_matrix = np.array(output_matrix)
#         return (output_matrix != 0).astype(int)
#     output_struct = binarize_matrix(output_struct)
#     assert type(output_struct) == np.ndarray, f"OUTPUT_STRUCT ERROR: The output structure is not a numpy array: {output_struct}"
    
#     function_class = evaluate_matrix(cfg, output_struct, function_class)

#     #logging.info(f"TASK {function_class.eval.idx_process} ---- final return")

#     return function_class

# # Find the index of the first non-zero element in the weight distribution
# def find_first_non_zero_index(lst):
#     for idx, item in enumerate(lst[1:], start=1):
#         if item != 0:
#             return idx
#     return None

# @fork
# def evaluate_matrix(cfg, output_struct, function_class: Type[FunctionClass]):
#     # from sage.all import matrix, LinearCode, GF, random_matrix, gap

#     # def gap_function():
#     #     while True:
#     #         try:
#     #             result = gap.version()
#     #             logging.info(f"Gap version: {result}")
#     #             dummy_code = LinearCode(random_matrix(GF(2), 12, 24))
#     #             dummy_distance = dummy_code.minimum_distance()
#     #             logging.info(f"TASK {function_class.eval.idx_process} dummy distance: {dummy_distance}")
#     #             break
#     #         except Exception as e:
#     #             print(f"GAP initialization error: {e}")
    
#     # gap_function()
#     # gap_function()
    
#     assert isinstance(output_struct, np.ndarray), f"ERROR: Output struct is not a np array: {output_struct}"
#     #function_class.eval.output_struct = output_struct

#     ## Postprocessing for matrix
#     # Get the final G_new matrix
#     if cfg.identity:
#         # Construct the identity matrix
#         Identity = np.eye(cfg.k, dtype=int)
#         G_new_numpy = np.concatenate((Identity, output_struct), axis=1)
#     else: 
#         G_new_numpy = output_struct

#     assert G_new_numpy.shape == (cfg.k, cfg.n), "ERROR: The shape of the matrix is not correct"
#     function_class.eval.G_new_numpy = G_new_numpy

#     # G = current_code.generator_matrix()
#     logging.info(f"TASK {function_class.eval.idx_process} evaluating the code")

#     # Remove for less strict
#     if cfg.self_ortho and (not cfg.less_strict):
#         if not orthogonality_check(G_new_numpy):
#             function_class.fail_flag = 1
#             function_class.fail.reason_orthogonality = 1
#             function_class.score = cfg.failed_score
#             function_class.true_score = cfg.failed_score
#             return function_class

#     if is_degenerate(G_new_numpy):
#         function_class.fail_flag = 1
#         function_class.fail.reason_degenerate = 1
#         function_class.score = cfg.failed_score
#         function_class.true_score = cfg.failed_score
#         return function_class
#     else:
#         G_new = matrix(GF(2), G_new_numpy)

#         code = LinearCode(G_new)

#         code_dimension = code.dimension()
#         code_dimension = int(code_dimension)

#         # if not cfg.score_weight_distr:
#         #     minimum_distance = code.minimum_distance()
#         #     # Convert Sage integers to Python integers
#         #     minimum_distance = int(minimum_distance)
#         #     logging.info(f"TASK {function_class.eval.idx_process} calculated distance {minimum_distance}")

#         #     # Dim penalty
#         #     dim_penalty = (2 * (cfg.k - code_dimension)) ** 2
#         #     dim_coeff = 1

#         #     logging.info(f"TASK {function_class.eval.idx_process} calculated dimension {code_dimension}")

#         #     # Weight penalty
#         #     # if cfg.weight_penalty:
#         #     #     weight_penalty = sum(code.weight_distribution()[1 : cfg.dist])  # 0.0
#         #     #     weight_penalty = int(weight_penalty)
#         #     # else:
#         #     #     weight_penalty = 0
#         #     # weight_coeff = 0.01

#         #     new_score = minimum_distance - dim_coeff * dim_penalty #- weight_coeff * weight_penalty
#         #     new_score = max(0, new_score)  # Cap the min score to 0
#         #     new_score = float(new_score)

#         # else:
#         weight_distribution = list(code.weight_distribution())
#         logging.info(f"TASK {function_class.eval.idx_process} calculated distr")
#         if cfg.score_debug:
#             new_score = - sum([weight_distribution[j] * cfg.score_multiply ** (j + 1) for j in range(1, cfg.dist)])
#             if cfg.n <= 40:
#                 new_score = 100 * new_score # multiply one more time to make the prompt easier to read
        
#         else:
#             sum_weight_distr = sum([weight_distribution[j] * 0.99 ** (j + 1) for j in range(1, cfg.dist)])
#             new_score = - np.log(sum_weight_distr + 1)
#             new_score = float(new_score)
#             new_score =  100 * round(new_score, 4) #round decimal places and scale

#         #minimum_distance = next((i for i, x in enumerate(weight_distribution) if x), None)
#         minimum_distance = int(find_first_non_zero_index(weight_distribution))

#         true_score = minimum_distance if code_dimension == cfg.k else -1
#         logging.info(f"TASK {function_class.eval.idx_process} final logging")
#         function_class.fail_flag = 0
#         function_class.correct_flag = 1 if code_dimension == cfg.k else 0
#         function_class.score = float(new_score) if code_dimension == cfg.k else cfg.failed_score
#         function_class.true_score = int(true_score)
#         function_class.eval.code_dimension = int(code_dimension)
#         function_class.eval.minimum_distance = int(minimum_distance)
#         #function_class.eval.dim_penalty = dim_penalty
#         #function_class.eval.weight_penalty = weight_penalty
#         #function_class.eval.applied_dim_penalty = dim_coeff * dim_penalty
#         #function_class.eval.applied_weight_penalty = weight_coeff * weight_penalty
#         #function_class.eval.weight_key = tuple(code.weight_distribution())  # Convert to tuple for hashing
#         # if cfg.calculate_canonical:
#         #     logging.info(f"TASK {function_class.eval.idx_process} calculating canonical form")
#         #     canonical_form = LinearCodeAutGroupCanLabel(code).get_canonical_form()
#         #     function_class.eval.canonical_key = canonical_form

#         logging.info(f"TASK {function_class.eval.idx_process} finished evaluation --- score: {new_score}")
#         return function_class

# if __name__== "__main__":
#     from omegaconf import OmegaConf
#     cfg = OmegaConf.create({
#         "identity":0,
#         "n": 256,
#         "k": 16,
#         "self_ortho": 0,
#         "weight_penalty": 0,
#         "score_weight_distr": 1,
#         "dist": 114,
#         "calculate_canonical": 0,
#     })

#     # Check time of final matrix evaluation
#     function_class = FunctionClass()
#     output_struct = np.array(random_matrix(GF(2), cfg.k, cfg.n))
#     import time
#     start_time = time.time()
#     evaluate_matrix(cfg, output_struct, function_class)
#     end_time = time.time()
#     print(f"Execution time = {int(end_time - start_time)} seconds")

#     #Check time of function execution
#     function_str = """
# def update_matrix(M):
#     n = M.shape[0]
#     # Initialize U as the identity matrix
#     U = np.eye(n, dtype=int)
#     # Perform random elementary row operations over GF(2)
#     for _ in range(n):
#         i = np.random.randint(0, n)
#         while i == n - 1 or (U[i, i] == 1 and not np.any(U[i, :n - 1] == 1)):
#             i = np.random.randint(0, n)
#         # Swap row i with row n-1
#         U[[i, n - 1]] = U[[n - 1, i]]
#         # Add row n-1 to row i modulo 2
#         U[i] = (U[i] + U[n - 1]) % 2
#         # Make row i nonzero
#         if np.all(U[i] == 0):
#             j = np.random.randint(0, n - 1)
#             U[i, j] = 1
#     # Multiply U with M over GF(2)
#     M_new = (U @ M) % 2
#     return M_new
#     """
#     function_class = FunctionClass(function_str=function_str, imports_str="")

#     input_struct = np.array(random_matrix(GF(2), cfg.k, cfg.n))
#     import time
#     start_time = time.time()
#     evaluate_func(cfg, input_struct, function_class)
#     end_time = time.time()
#     print(f"Execution time = {int(end_time - start_time)} seconds")


# # if __name__ == "__main__":

# #     # Parameters for the binary code search
# #     n = 300  # Length of the code (for example, dimension 24 for Golay code)
# #     k = 30  # Dimension of the code (adjust as necessary)
# #     seed = 0

# #     cfg = Namespace(
# #         n=n,
# #         k=k,
# #         seed=seed,
# #         generate_poly_coeff=False,
# #         identity=True,
# #         sparse_representation=False,
# #         apply_while_different=False,
# #         apply_10_times=False,
# #         weight_penalty=False,
# #         dist=8,
# #         function_str_to_extract="perturb_matrix",
# #         start_with_zero=False,
# #         start_with_one=False,
# #         idx_process=-1,
# #     )

# #     seed_everything(cfg.seed)
# #     input_struct = generate_input(cfg)

# #     function_str = """
# # def perturb_matrix(M):
# #     M_new = np.copy(M)
# #     num_rows, num_cols = M.shape

# #     # Determine which element to flip by iterating over the matrix in a specific order
# #     for i in range(num_rows):
# #         for j in range(num_cols):
# #             M_new[i, j] = 1

# #     return M_new"""

# #     function_class = FunctionClass(function_str=function_str, imports_str="")

# #     print(f"Evaluating n = {cfg.n}, k = {cfg.k}, d = {cfg.dist}")
# #     # Time the execution
# #     start_time = time.time()
# #     results = evaluate_func(cfg, input_struct, function_class)
# #     end_time = time.time()
# #     print(f"Execution time = {int(end_time - start_time)} seconds")
# #     print(f"Failed? {results.fail_flag}")
# #     print(f"Score: {results.score}")
# #     print(f"True distance: {results.true_score}")

# #     def construct_cyclic_generator_matrix(row):
# #         assert isinstance(row, np.ndarray)
# #         row = list(row)
# #         # Correct initial row based on the generator polynomial
# #         row = row + [0] * 11

# #         # Create the generator matrix
# #         G = np.zeros((12, 23), dtype=int)

# #         for i in range(12):
# #             G[i] = np.roll(row, -i)
# #         return G

# #     # row = np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1])
# #     # row = np.array([1] * 12)

# #     # Generate and print the matrix
# #     # G = golay_generator_matrix(row)
# #     # G_sage = matrix(GF(2), G)
# #     # print(LinearCode(G_sage).minimum_distance())

# #     # print(2**12)

# #     # def perturb_matrix_v2(M):
# #     #     """
# #     #     Perturbs the generator matrix M by flipping a bit in a controlled way.

# #     #     Args:
# #     #         M (numpy array): Matrix over GF(2)

# #     #     Returns:
# #     #         M_new (numpy array): Perturbed matrix
# #     #     """
# #     #     M_new = np.copy(M)
# #     #     num_rows, num_cols = M.shape

# #     #     # Determine which element to flip by iterating over the matrix in a specific order
# #     #     for i in range(num_rows):
# #     #         for j in range(num_cols):
# #     #             if M[i, j] == 0:  # Flip elements that are 0
# #     #                 M_new[i, j] = 1

# #     #     return M_new

# #     # def perturb_matrix_v2(M):
# #     #     """
# #     #     Perturbs the generator matrix M by flipping two bits.

# #     #     Args:
# #     #         M (numpy array): Matrix over GF(2)

# #     #     Returns:
# #     #         M_new (numpy array): Perturbed matrix
# #     #     """
# #     #     M_new = np.copy(M)
# #     #     n, k = M_new.shape
# #     #     stop = min(n, k)
# #     #     for i in range(stop):
# #     #         M_new[i, i] = 1 - M[i, i]
# #     #     return M_new

# #     # def perturb_matrix_v2(M):
# #     #     """
# #     #     Perturbs the generator matrix M by flipping a bit and swapping rows.

# #     #     Args:
# #     #         M (numpy array): Matrix over GF(2)

# #     #     Returns:
# #     #         M_new (numpy array): Perturbed matrix
# #     #     """
# #     #     M_new = np.copy(M)
# #     #     M_new[5, 5] = 1 - M[5, 5]
# #     #     if M_new[0, 1] == 1:
# #     #         M_new[[0, 5]] = M_new[[5, 0]]
# #     #     return M_new

# #     # def perturb_matrix(M):
# #     #     """
# #     #     Perturbs the generator matrix M by flipping a bit, swapping columns, and flipping a row.

# #     #     Args:
# #     #         M (numpy array): Matrix over GF(2)

# #     #     Returns:
# #     #         M_new (numpy array): Perturbed matrix of the same shape as M
# #     #     """
# #     #     M_new = M.copy()  # Create a copy of M

# #     #     # Flip a bit: Start from the top left corner and move towards the bottom right.
# #     #     # When we encounter a 1, flip it to 0 and when we encounter a 0, stop.
# #     #     for i in range(M.shape[0]):
# #     #         for j in range(M.shape[1]):
# #     #             if M_new[i, j] == 1:
# #     #                 M_new[i, j] = 0
# #     #                 break
# #     #         if i == M.shape[0] - 1:
# #     #             break

# #     #     # Swap the last column with a random column
# #     #     col_idx = np.random.randint(0, M.shape[1] - 1)
# #     #     M_new[:, col_idx], M_new[:, M.shape[1] - 1] = M_new[:, M.shape[1] - 1], M_new[:, col_idx]

# #     #     # Flip a row: Start from the top row and move towards the bottom.
# #     #     # When we encounter all 1s, stop.
# #     #     for i in range(M.shape[1]):
# #     #         for j in range(M.shape[0]):
# #     #             if M_new[j, i] == 0:
# #     #                 M_new[j, i] = 1
# #     #                 break
# #     #         if i == M.shape[1] - 1:
# #     #             break

# #     #     return M_new

# #     # def perturb_matrix_v2(M):
# #     #     """
# #     #     Perturbs the generator matrix M by swapping rows, flipping bits, and adding a new column deterministically.

# #     #     Args:
# #     #         M (numpy array): Matrix over GF(2)

# #     #     Returns:
# #     #         M_new (numpy array): Perturbed matrix
# #     #     """
# #     #     M_new = np.copy(M)

# #     #     # Swap the row with the most 1s with the second-to-last row
# #     #     swap_row = np.argmax((M_new == 1).sum(axis=1)) + 1
# #     #     M_new[[swap_row, -2]] = M_new[[(-2), swap_row]].copy()

# #     #     # Flip the columns with the largest number of 1s
# #     #     flip_cols = np.arange(M_new.shape[1])[np.argsort(-(M_new == 1).sum(axis=0))[:M_new.shape[1]//2]]
# #     #     M_new[:, flip_cols] = 1 - M_new[:, flip_cols]

# #     #     # Add a new column with the most frequent bit
# #     #     ones = (M_new == 1).sum()
# #     #     zeros = M_new.size - ones
# #     #     if ones >= zeros:
# #     #         M_new = np.hstack((M_new, np.ones((M_new.shape[0], 1))))
# #     #     else:
# #     #         M_new = np.hstack((M_new, np.zeros((M_new.shape[0], 1))))

# #     #     return M_new

# #     # input = generate_input(args)

# #     # print("Initial code:")
# #     # print(input)

# #     # G_new_numpy = perturb_matrix_func(G_numpy)

# #     # score, score_dummy = evaluate_func(input, n, None, None, perturb_matrix)
# #     # print(score, score_dummy)


# # # def perturb_matrix(G):
# # #     """Perturbs the generator matrix by flipping a bit or swapping rows."""
# # #     new_G = copy(G)
# # #     if random.random() < 0.5:
# # #         i = random.randint(0, G.nrows() - 1)
# # #         j = random.randint(0, G.ncols() - 1)
# # #         new_G[i, j] = 1 - G[i, j]
# # #     else:
# # #         i, j = random.sample(range(G.nrows()), 2)
# # #         new_G.swap_rows(i, j)
# # #     return new_G
