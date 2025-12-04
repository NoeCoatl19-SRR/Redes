import numpy as np

sizes=[5,2,1]
biases = [np.random.normal(0, 1/np.sqrt(y), (y, 1)) for y in sizes[1:]]
weights = [np.random.normal(0, 1/np.sqrt(x), (y, x)) for x, y in zip(sizes[:-1], sizes[1:])]

print(sizes)
print(biases)
print(weights)
print("eh hola amigo")