import numpy as np
import matplotlib.pyplot as plt
import copy


def l2Norm(a, b):
    return (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def l2NormNoSqrt(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


k = 6
r = 10
m = 1.1
data = np.genfromtxt("cluster_dataset.txt", delimiter="  ")
rowCount = data.shape[0]
columnCount = data.shape[1]

finalSSE = float("inf")
finalIterations = []
for i in range(r):
    sse = float("inf")
    # generate random intial membership grades
    memGrades = np.zeros((rowCount, k))
    for i in range(rowCount):
        for j in range(k):
            memGrades[i][j] = np.random.uniform(0, 1)
        rowSum = sum(memGrades[i])
        memGrades[i] = memGrades[i] / rowSum

    iterations = []
    improving = True
    while improving:
        # calculate new centroids
        centroids = np.sum(data * np.transpose((memGrades ** m))[:, :, None], axis=1) / np.sum(np.transpose(memGrades) ** m, axis=1)[:, np.newaxis]

        # calculate new membership grades
        for i in range(rowCount):
            for j in range(k):
                numerator = l2Norm(data[i], centroids[j])
                totalSum = 0
                for l in range(k):
                    totalSum += (numerator / l2Norm(data[i], centroids[l])) ** (2 / (m - 1))
                memGrades[i][j] = 1 / totalSum

        # calculate sse
        sseTotal = 0
        for i in range(rowCount):
            for j in range(k):
                sseTotal += (memGrades[i][j] ** m) * l2NormNoSqrt(data[i], centroids[j])
        if sse - sseTotal < 0.1 and not(sse == float("inf")):
            improving = False
        sse = sseTotal

        assignedCluster = np.argmax(memGrades, axis=1)
        # split up data based on cluster
        dataGroups = []
        for i in range(k):
            dataGroups.append(data[assignedCluster == i])
        dataGroups = np.asarray(dataGroups)
        # save all iterations of the centroids moving
        iterations.append((centroids, dataGroups))
        # print(sseTotal)

    if finalSSE > sse:
        finalSSE = sse
        finalIterations = copy.deepcopy(iterations)

print(finalSSE)
# print best iteration group
for i in range(len(finalIterations)):
    for j in range(k):
        color = [[(j % k) / k, (j * 2 % k) / k, (j * 3 % k) / k]]
        plt.scatter(finalIterations[i][1][j][:, 0], finalIterations[i][1][j][:, 1], c=color)

    crosses = plt.scatter(finalIterations[i][0][:, 0], finalIterations[i][0][:, 1], c="red", marker="X")
    plt.legend([crosses], ["centroids"], scatterpoints=1, loc='lower right')
    plt.savefig("outputs/output" + str(i))
    plt.clf()