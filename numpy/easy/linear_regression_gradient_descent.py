import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	m, n = X.shape
	theta = np.zeros((n, 1))

    for i in range(iterations):
        y_hat = X @ theta
        error = (y_hat - y.reshape(-1,1))
        gradient = (1/m) * X.T @ error
        theta = theta - alpha * gradient

	return np.round(theta, 4)
