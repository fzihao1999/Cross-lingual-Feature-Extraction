import numpy as np

def min_max(path):
    with open(path, 'r', encoding='utf8') as file:
        header = file.readline().split(' ')
        count = int(header[0])
        dim = int(header[1])
        matrix = np.empty((count, dim))
        for i in range(count):
            word, vec = file.readline().split(' ', 1)
            matrix[i] = np.fromstring(vec, sep=' ')
        maxtrix = (matrix - np.mean(matrix, axis=1).reshape(count, 1)) / (np.max(matrix, axis=1) - np.min(matrix, axis=1)).reshape(count, 1)
    

if __name__ == "__main__":
    