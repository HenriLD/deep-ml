import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	
	layer1 = w1 @ x
	layer2 = w2 @ relu(layer1)
	residual = layer2 + x
	output = relu(residual)
	
	return output

def relu(x: np.ndarray) -> np.ndarray:
	return np.where(x > 0, x, 0)
