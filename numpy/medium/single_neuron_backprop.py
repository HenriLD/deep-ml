import numpy as np

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []

    for i in range(epochs):
        y_hat = features @ updated_weights + updated_bias
        prediction = 1 / (1 + np.exp(-y_hat))

        loss = np.mean((prediction - labels)**2)
        mse_values.append(loss)

        error = prediction - labels
        sigmoid_derivative = prediction * (1 - prediction)
        gradient_w = (2 * error * sigmoid_derivative) @ features / features.shape[0]
        gradient_b = np.mean(2 * error * sigmoid_derivative)

        updated_weights -= learning_rate * gradient_w
        updated_bias -= learning_rate * gradient_b

	return updated_weights, updated_bias, mse_values
