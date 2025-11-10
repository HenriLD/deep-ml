import numpy as np

def overlapping_max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Applies overlapping max pooling to a 4D tensor (N, C, H, W).

    Args:
        x: Input array of shape (N, C, H, W)
        kernel_size: Size of pooling window (int)
        stride: Stride between pooling windows (int)

    Returns:
        A 4D tensor after overlapping pooling.
    """
    N, C, H, W = x.shape

    out_h = (H - kernel_size) // stride + 1
    out_w = (W - kernel_size) // stride + 1
    output_matrix = np.zeros((N, C, out_h, out_w))

    for y in range(out_h):
        for z in range(out_w):
            patch = x[:, :, y * stride: y * stride + kernel_size, z * stride: z * stride + kernel_size]
            output_matrix[:, :, y,z] = np.max(patch, axis=(2,3))

    return output_matrix
