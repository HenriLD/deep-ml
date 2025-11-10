import numpy as np

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

    if mode == 'row':
        axis = 1
    else:
        axis = 0
        
    array = np.array(matrix)

	return np.mean(array, axis=axis)
