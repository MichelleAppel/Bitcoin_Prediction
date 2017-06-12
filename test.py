import numpy as np

list1 = np.array([[0], [1], [3], [4], [5]])
list1 = list1.reshape(np.size(list1))
list2 = np.array([1, 1, 3, 4, 5])

print(list1 - list2)