import numpy as np

def matrixmul(a:list[list[int|float]],
              b:list[list[int|float]])-> list[list[int|float]]:

    A = np.array(a)
    B = np.array(b)

    try:
        return A @ B
	except:
        return -1
