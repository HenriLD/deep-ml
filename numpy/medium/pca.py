import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	
    standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    covariance_matrix = np.cov(standardized_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    principal_components = sorted_eigenvectors[:, :k]

	return np.round(principal_components, 4)
