import numpy as np
import matplotlib.pyplot as plt
import copy

k = 6
r = 10
data = np.genfromtxt("cluster_dataset.txt", delimiter="  ")
rowCount = data.shape[0]
columnCount = data.shape[1]
maxRand = np.amax(data)
minRand = np.amin(data)

sse = float("inf")
for i in range(r):
    # generate random centroids
    centroids = np.zeros((k, 2))
    for i in range(k):
        duplicate = True
        while duplicate:
            centroids[i][0] = np.random.uniform(minRand, maxRand)
            centroids[i][1] = np.random.uniform(minRand, maxRand)
            duplicate = False
            for j in range(i):
                if np.array_equal(centroids[j], centroids[i]):
                    duplicate = True
                    break

    # array to keep track of what cluster the corresponding data is in
    assignedCluster = np.zeros(rowCount, dtype=int)

    changes = 1
    iterations = []
    while changes != 0:
        # changes = False
        changes = 0
        # assign each data point to a cluster
        for i in range(rowCount):
            idx = assignedCluster[i]
            minDistance = (data[i][0] - centroids[idx][0]) ** 2 + (data[i][1] - centroids[idx][1]) ** 2
            for j in range(0, k):
                distance = (data[i][0] - centroids[j][0]) ** 2 + (data[i][1] - centroids[j][1]) ** 2
                if distance < minDistance:
                    minDistance = distance
                    assignedCluster[i] = j
                    changes += 1

        # find new means
        if changes != 0:
            dataCount = np.ones(k)
            newCentroids = np.zeros((k, 2))
            for i in range(rowCount):
                idx = assignedCluster[i]
                dataCount[idx] += 1
                newCentroids[idx] += data[i]
            for j in range(k):
                newCentroids[j] = newCentroids[j] / dataCount[j]
            centroids = newCentroids

        # split up data based on cluster
        dataGroups = []
        for i in range(k):
            dataGroups.append(data[assignedCluster == i])
        dataGroups = np.asarray(dataGroups)
        # save all iterations of the centroids moving
        iterations.append((centroids, dataGroups))

    # calculate sum of sqaures error
    tempSum = 0
    for i in range(k):
        tempSum += sum((dataGroups[i][:, 0] - newCentroids[i][0]) ** 2 + (dataGroups[i][:, 1] - newCentroids[i][1]) ** 2)
    if tempSum < sse:
        # sse
        sse = tempSum

        finalIterations = copy.deepcopy(iterations)

# print best iteration group
for i in range(len(finalIterations)):
    for j in range(k):
        color = [[(j % k) / k, (j * 2 % k) / k, (j * 3 % k) / k]]
        plt.scatter(finalIterations[i][1][j][:, 0], finalIterations[i][1][j][:, 1], c=color)

    crosses = plt.scatter(finalIterations[i][0][:, 0], finalIterations[i][0][:, 1], c="red", marker="X")
    plt.legend([crosses], ["centroids"], scatterpoints=1, loc='lower right')
    plt.savefig("outputs/output" + str(i))
    plt.clf()
