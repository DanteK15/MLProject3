import numpy as np
import matplotlib.pyplot as plt
import copy


def l2Norm(a, b):
    return (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def l2NormNoSqrt(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


k = 3
r = 1
m = 2
data = np.genfromtxt("cluster_dataset.txt", delimiter="  ")
rowCount = data.shape[0]
columnCount = data.shape[1]
maxRand = np.amax(data)
minRand = np.amin(data)

for i in range(r):
    sse = float("inf")
    # generate random intial membership grades
    memGrades = np.zeros((rowCount, k))
    for i in range(rowCount):
        for j in range(k):
            memGrades[i][j] = np.random.uniform(1, 10)
        rowSum = sum(memGrades[i])
        memGrades[i] = memGrades[i] / rowSum

    improving = True
    while improving:
        # example he gave
        """
        data = np.array([[1, 2], [0, -1]])
        rowCount = 2
        memGrades = np.array([[0.4, 0.6], [0.7, 0.3]])
        k = 2
        m = 2
        """

        # calculate new centroids
        centroids = np.sum(data * np.transpose((memGrades ** m))[:, :, None], axis=1) / np.sum(np.transpose(memGrades) ** m, axis=1)[:, np.newaxis]

        # print(centroids)
        # calculate new membership grades
        for i in range(rowCount):
            for j in range(k):
                numerator = l2Norm(data[i], centroids[j])
                totalSum = 0
                for l in range(k):
                    totalSum += (numerator / l2Norm(data[i], centroids[l])) ** (2 / (m - 1))
                memGrades[i][j] = 1 / totalSum
        # print(memGrades)

        # calculate sse
        sseTotal = 0
        for i in range(rowCount):
            for j in range(k):
                sseTotal += (memGrades[i][j] ** m) * l2NormNoSqrt(data[i], centroids[j])
        if sse - sseTotal < 0.1 and not(sse == float("inf")):
            improving = False
        sse = sseTotal
        print(sseTotal)
