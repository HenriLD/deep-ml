import math

def softmax(scores: list[float]) -> list[float]:
    
    probabilities = []
    cumsum = 0

	for number in scores:
        probabilities.append(math.exp(number))
        cumsum += math.exp(number)

    for i in range(len(probabilities)):
        probabilities[i] = probabilities[i] / cumsum

	return probabilities
