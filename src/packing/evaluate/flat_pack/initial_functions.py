import numpy as np


def all_equal(
        current_grid: np.ndarray,
        blocks: np.ndarray,
        action_mask: np.ndarray
) -> np.ndarray:
    """
    Computes heuristic Q-values all equal for all valid (block, rotation, row, col) placements.

    Args:
        current_grid: Grid representing the current state of the environment.
        blocks: Array containing available blocks.
        action_mask: Boolean mask indicating valid placements.

    Returns:
        A Q-value matrix of shape (num_blocks, 4, num_rows, num_cols).
    """
    # Precompute rotated versions of all blocks
    num_blocks = blocks.shape[0]
    rotated_blocks = np.array([[np.rot90(block, k=r) for r in range(4, 0, -1)] for block in blocks])

    # Pad the grid once (for boundary checking)
    padded_grid = np.pad(current_grid, 1, mode='constant', constant_values=0)

    # Initialize Q-value matrix
    q_values = np.full(action_mask.shape, -np.inf)

    # Vectorized placement loop to
    for block_idx in range(num_blocks):
        for rotation in range(4):
            block = rotated_blocks[block_idx, rotation]
            block_rows, block_cols = block.shape

            # Extract all possible placements using NumPy slicing
            sub_grids = np.lib.stride_tricks.sliding_window_view(padded_grid, (block_rows, block_cols))

            # Compute final scores, with the stride trick above it becomes possible to efficiently assign scores over
            # all possible placements. For now just assigns equal score everywhere.
            q_values[block_idx, rotation, ...] = 1

    # Apply action mask in one operation
    q_values[~action_mask] = -np.inf

    return q_values


def heuristic_flatpack(
        current_grid: np.ndarray,
        blocks: np.ndarray,
        action_mask: np.ndarray
) -> np.ndarray:
    """
    Computes heuristic Q-values for all valid (block, rotation, row, col) placements.

    Args:
        current_grid: Grid representing the current state of the environment.
        blocks: Array containing available blocks.
        action_mask: Boolean mask indicating valid placements.

    Returns:
        A Q-value matrix of shape (num_blocks, 4, num_rows, num_cols).
    """
    # Precompute rotated versions of all blocks
    num_blocks = blocks.shape[0]
    rotated_blocks = np.array([[np.rot90(block, k=r) for r in range(4)] for block in blocks])

    # Precompute block sizes
    block_sizes = np.sum(rotated_blocks[:, 0] > 0, axis=(1, 2))  # Only compute for unrotated block

    # Pad the grid once (for boundary checking)
    padded_grid = np.pad(current_grid, 1, mode='constant', constant_values=0)

    # Initialize Q-value matrix
    q_values = np.full(action_mask.shape, -np.inf)

    # Vectorized placement loop
    for block_idx in range(num_blocks):
        for rotation in range(4):
            block = rotated_blocks[block_idx, rotation]
            block_rows, block_cols = block.shape

            # Extract all possible placements using NumPy slicing
            sub_grids = np.lib.stride_tricks.sliding_window_view(padded_grid, (block_rows, block_cols))

            # Compute adjacency scores using broadcasting
            adjacency_scores = np.sum((sub_grids > 0) & (block > 0), axis=(-2, -1))

            # Compute isolation penalties
            isolation_penalties = np.where(adjacency_scores > 0, 0, -block_sizes[block_idx])

            # Compute final scores
            q_values[block_idx, rotation, ...] = (
                    block_sizes[block_idx] + adjacency_scores[:-2, :-2] + isolation_penalties[:-2, :-2]
            )

    # Apply action mask in one operation
    q_values[~action_mask] = -np.inf

    return q_values


def random_q_function(
        current_grid: np.ndarray,
        blocks: np.ndarray,
        action_mask: np.ndarray
) -> np.ndarray:
    """
    Generates completely random Q-values for all (block, rotation, row, col) placements.

    Args:
        current_grid: Grid representing the current state of the environment.
        blocks: Array containing available blocks.
        action_mask: Boolean mask indicating valid placements.

    Returns:
        A Q-value matrix of shape (num_blocks, 4, num_rows, num_cols).
    """
    # Generate random scores between 0 and 1
    q_values = np.random.rand(*action_mask.shape)

    return q_values
