import numpy as np
import matplotlib.pyplot as plt
import copy


def l2Norm(a, b):
    return (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def l2NormNoSqrt(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


k = 3
r = 10
m = 2
data = np.genfromtxt("Programming 3/cluster_dataset.txt", delimiter="  ")
rowCount = data.shape[0]
columnCount = data.shape[1]
maxRand = np.amax(data)
minRand = np.amin(data)

sse = float("inf")
for i in range(r):
    # generate random intial membership grades
    memGrades = np.zeros((rowCount, k))
    for i in range(rowCount):
        for j in range(k):
            memGrades[i][j] = np.random.uniform(1, 10)
        rowSum = sum(memGrades[i])
        memGrades[i] = memGrades[i] / rowSum

    for j in range(1):
        # example he gave
        """
        data = np.array([[1, 2], [0, -1]])
        rowCount = 2
        memGrades = np.array([[0.4, 0.6], [0.7, 0.3]])
        k = 2
        m = 1
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
        sse = 0
        for i in range(rowCount):
            for j in range(k):
                sse += (memGrades[i][j] ** m) * l2NormNoSqrt(data[i], centroids[j])
        # print(sse)
