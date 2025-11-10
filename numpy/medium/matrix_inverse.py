import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:

	A = np.array(matrix)
	
	if np.linalg.det(A) == 0:
		return None

	return np.linalg.inv(A)
