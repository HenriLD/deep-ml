import numpy as np

def cosine_similarity(v1, v2):
	
	return round((np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))), ndigits = 3)
