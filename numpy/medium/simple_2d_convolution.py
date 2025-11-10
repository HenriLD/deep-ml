import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	padded_matrix = np.pad(input_matrix, padding, mode='constant')

    out_height = ((input_height + 2 * padding - kernel_height) // stride) + 1
    out_width = ((input_width + 2 * padding - kernel_width) // stride) + 1

    output_matrix = np.zeros((out_height, out_width))

    for y in range(out_height):
        for x in range(out_width):
            
            h_start = y * stride
            w_start = x * stride
            h_end = h_start + kernel_height
            w_end = w_start + kernel_width
            patch = padded_matrix[h_start:h_end, w_start:w_end]
            output_matrix[y, x] = (patch * kernel).sum()

	return output_matrix
