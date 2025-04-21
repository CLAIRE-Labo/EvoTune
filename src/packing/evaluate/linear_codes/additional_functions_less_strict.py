import numpy as np
import random
from packing.logging.function_class import FunctionClass
from packing.utils.functions import function_to_string

def update_matrix(M):
    """
    Updates the input self-orthogonal binary matrix M and returns a new binary 
    self-orthogonal matrix M_new.

    The function constructs an orthogonal multiplier matrix over GF(2) and multiplies
    it with the input matrix M to produce a new self-orthogonal matrix M_new.

    Args:
        M (numpy.ndarray): A self-orthogonal binary matrix of shape (k, n).

    Returns:
        numpy.ndarray: A new self-orthogonal binary matrix of shape (k, n).

    Raises:
        AssertionError: If the resulting matrix is not self-orthogonal.

    Notes:
        - A matrix M_new is self-orthogonal if (M_new @ M_new.T) % 2 == 0.
        - The multiplier matrix is constructed to be orthogonal over GF(2), i.e.,
        it satisfies (C @ C.T) % 2 == I, where I is the identity matrix.
        - Multiplying M by an orthogonal matrix preserves the self-orthogonality of M.
    """
    k, n = M.shape

    # Select four unique random column indices
    selected_indices = random.sample(range(n), 8)
    
    # Initialize a binary multiplier matrix with zeros
    C = np.zeros((n, n), dtype=int)
    
    # Set ones in the selected positions
    for i in selected_indices:
        for j in selected_indices:
            C[i, j] = 1
    
    # Add the identity matrix modulo 2 to create an orthogonal matrix
    C = (C + np.identity(n, dtype=int)) % 2

    # Multiply the multiplier matrix with the input matrix and apply modulo 2
    M_new = (M @ C) % 2

    return M_new

function_class1 = FunctionClass(function_to_string(update_matrix), "import random\nimport numpy as np")


# def sample_binary_orthogonal_matrix(n):
#     """
#     Generates a random binary orthogonal matrix of size n x n.
    
#     Parameters:
#     n (int): The size of the matrix.
    
#     Returns:
#     numpy.ndarray: An n x n binary orthogonal (permutation) matrix.
#     """
#     # Generate a random permutation of the indices
#     permutation = np.random.permutation(n)
#     # Create an n x n zero matrix
#     P = np.zeros((n, n), dtype=int)
#     # Place 1s at the positions specified by the permutation
#     P[np.arange(n), permutation] = 1
#     return P


def update_matrix(M):
    """
    Updates the input self-orthogonal binary matrix M and returns a new binary 
    self-orthogonal matrix M_new.

    The function constructs a random permutation matrix over GF(2) and multiplies
    it with the input matrix M to produce a new self-orthogonal matrix M_new.

    Args:
        M (numpy.ndarray): A self-orthogonal binary matrix of shape (k, n).

    Returns:
        numpy.ndarray: A new self-orthogonal binary matrix of shape (k, n).

    Raises:
        AssertionError: If the resulting matrix is not self-orthogonal.

    Notes:
        - A matrix M_new is self-orthogonal if (M_new @ M_new.T) % 2 == 0.
        - The multiplier matrix is constructed as a permutation matrix, which is orthogonal over GF(2), i.e.,
          it satisfies (P @ P.T) % 2 == I, where I is the identity matrix.
        - Multiplying M by an orthogonal matrix preserves the self-orthogonality of M.
    """
    k, n = M.shape

    # Generate a random permutation of the columns
    permutation = np.random.permutation(n)
    
    # Create a permutation matrix P based on the permutation
    P = np.zeros((n, n), dtype=int)
    P[np.arange(n), permutation] = 1

    # Multiply the input matrix M with the permutation matrix P
    M_new = (M @ P) % 2

    return M_new

function_class2 = FunctionClass(function_to_string(update_matrix), "import numpy as np")


def update_matrix(M):
    """
    Updates the input self-orthogonal binary matrix M and returns a new binary 
    self-orthogonal matrix M_new.

    The function constructs orthogonal multiplier matrices over GF(2) multiple times and multiplies
    them with the input matrix M to produce a new self-orthogonal matrix M_new.

    Args:
        M (numpy.ndarray): A self-orthogonal binary matrix of shape (k, n).

    Returns:
        numpy.ndarray: A new self-orthogonal binary matrix of shape (k, n).

    Raises:
        AssertionError: If the resulting matrix is not self-orthogonal.

    Notes:
        - A matrix M_new is self-orthogonal if (M_new @ M_new.T) % 2 == 0.
        - The multiplier matrices are constructed to be orthogonal over GF(2), i.e.,
          they satisfy (C @ C.T) % 2 == I, where I is the identity matrix.
        - Multiplying M by an orthogonal matrix preserves the self-orthogonality of M.
        - This process is repeated multiple times with freshly sampled multiplier matrices.
    """
    import numpy as np
    import random

    k, n = M.shape
    num_iterations = 5  # Specify how many times to repeat the process

    M_new = M.copy()
    for _ in range(num_iterations):
        # Select eight unique random column indices
        selected_indices = random.sample(range(n), 8)
        
        # Initialize a binary multiplier matrix with zeros
        C = np.zeros((n, n), dtype=int)
        
        # Set ones in the selected positions
        for i in selected_indices:
            for j in selected_indices:
                C[i, j] = 1
        
        # Add the identity matrix modulo 2 to create an orthogonal matrix
        C = (C + np.identity(n, dtype=int)) % 2
        
        # Multiply the multiplier matrix with the input matrix and apply modulo 2
        M_new = (M_new @ C) % 2

    return M_new

function_class3 = FunctionClass(function_to_string(update_matrix), "import numpy as np")

def update_matrix(M):
    """
    Updates the input self-orthogonal binary matrix M and returns a new binary 
    self-orthogonal matrix M_new.

    The function constructs orthogonal multiplier matrices over GF(2) multiple times and multiplies
    them with the input matrix M to produce a new self-orthogonal matrix M_new.

    Args:
        M (numpy.ndarray): A self-orthogonal binary matrix of shape (k, n).

    Returns:
        numpy.ndarray: A new self-orthogonal binary matrix of shape (k, n).

    Raises:
        AssertionError: If the resulting matrix is not self-orthogonal.

    Notes:
        - A matrix M_new is self-orthogonal if (M_new @ M_new.T) % 2 == 0.
        - The multiplier matrices are constructed to be orthogonal over GF(2), i.e.,
          they satisfy (C @ C.T) % 2 == I, where I is the identity matrix.
        - Multiplying M by an orthogonal matrix preserves the self-orthogonality of M.
        - This process is repeated multiple times with freshly sampled multiplier matrices.
    """

    k, n = M.shape
    num_iterations = 20  # Specify how many times to repeat the process

    M_new = M.copy()
    for _ in range(num_iterations):
        # Select eight unique random column indices
        selected_indices = random.sample(range(n), 8)
        
        # Initialize a binary multiplier matrix with zeros
        C = np.zeros((n, n), dtype=int)
        
        # Set ones in the selected positions
        for i in selected_indices:
            for j in selected_indices:
                C[i, j] = 1
        
        # Add the identity matrix modulo 2 to create an orthogonal matrix
        C = (C + np.identity(n, dtype=int)) % 2
        
        # Multiply the multiplier matrix with the input matrix and apply modulo 2
        M_new = (M_new @ C) % 2

    return M_new

function_class4 = FunctionClass(function_to_string(update_matrix), "import numpy as np")



def update_matrix(M):
    """
    Updates the input self-orthogonal binary matrix M and returns a new binary 
    self-orthogonal matrix M_new.

    The function constructs an orthogonal multiplier matrix over GF(2) and multiplies
    it with the input matrix M to produce a new self-orthogonal matrix M_new.

    Args:
        M (numpy.ndarray): A self-orthogonal binary matrix of shape (k, n).

    Returns:
        numpy.ndarray: A new self-orthogonal binary matrix of shape (k, n).

    Raises:
        AssertionError: If the resulting matrix is not self-orthogonal.

    Notes:
        - A matrix M_new is self-orthogonal if (M_new @ M_new.T) % 2 == 0.
        - The multiplier matrix is constructed to be orthogonal over GF(2), i.e.,
        it satisfies (C @ C.T) % 2 == I, where I is the identity matrix.
        - Multiplying M by an orthogonal matrix preserves the self-orthogonality of M.
    """
    k, n = M.shape

    # Select four unique random column indices
    selected_indices = random.sample(range(n), 4)
    
    # Initialize a binary multiplier matrix with zeros
    C = np.zeros((n, n), dtype=int)
    
    # Set ones in the selected positions
    for i in selected_indices:
        for j in selected_indices:
            C[i, j] = 1
    
    # Add the identity matrix modulo 2 to create an orthogonal matrix
    C = (C + np.identity(n, dtype=int)) % 2

    # Multiply the multiplier matrix with the input matrix and apply modulo 2
    M_new = (M @ C) % 2

    return M_new

original_function = FunctionClass(function_to_string(update_matrix), "import random\nimport numpy as np")


# selected_indices = [2, 5]
# C = np.zeros((9, 9), dtype=int)
# for i in selected_indices:
#     for j in selected_indices:
#         C[i, j] = 1

# C = (C + np.identity(9, dtype=int)) #% 2

# C @ C.T %2

def update_matrix(M):
    """
    Updates the input binary matrix M and returns a new binary matrix M_new.

    Args:
        M (numpy.ndarray): A 2D numpy array where each element is either 0 or 1.
                        This represents the input binary matrix to be updated.

    Returns:
        M_new (numpy.ndarray): A 2D numpy array of the same shape as M, where each element is either 0 or 1.
                    This represents the updated binary matrix. Currently, it is identical to the input matrix M.

    Note:
        This is a placeholder function that currently does not modify the input matrix.
        In a real implementation, this function would contain logic to update the matrix based on specific rules or conditions.
        Replace the placeholder logic with the actual matrix update logic as required.
    """
    M_new = M
    return M_new

identity_function = FunctionClass(function_to_string(update_matrix), "")

def update_matrix(M):
    """
    Perturbs the input binary matrix M my multiplying it with a random invertible matrix.

    This function generates a random invertible binary matrix U over GF(2) by performing random elementary
    row operations starting from the identity matrix. It then multiplies U with M over GF(2) to produce
    the updated matrix M_new.

    Args:
        M (numpy.ndarray): A 2D numpy array with elements 0 or 1, representing the input binary matrix.

    Returns:
        numpy.ndarray: A 2D numpy array of the same shape as M, containing the updated binary matrix. Each element is either 0 or 1.
    """
    n = M.shape[0]
    # Initialize U as the identity matrix
    U = np.eye(n, dtype=int)
    # Perform random elementary row operations over GF(2)
    for _ in range(n * n):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            # Add row j to row i modulo 2
            U[i] = (U[i] + U[j]) % 2
    # Multiply U with M over GF(2)
    M_new = (U @ M) % 2
    return M_new

random_function = FunctionClass(function_to_string(update_matrix), "import numpy as np")