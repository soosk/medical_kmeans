import numpy as np
import matplotlib.pyplot as plt
from random import randint
from csv import reader
from collections import defaultdict
from operator import itemgetter


def average(points):
    """
    Given a list of n-dimensional points return a point as an average
    """
    return np.array(points).mean(axis = 0)

def dist(a, b):
    """
    find the euclidean distance between two points
    """
    return np.linalg.norm(np.array(a)-np.array(b))

def loadData():
    """
    load the data we want to work with. the last index in each row corresponds to
    the label of the group (0 or 1). We will disregard that last element in our processing
    """
    data = []
    with open('dataset.txt') as csvfile:
        scanner = reader(csvfile)
        count = 0
        for row in scanner:
            #Features:   0:age | 3: resting blood pressure | 4: chol | 7: maximum heart rate achieved  | 9: oldpeak | 10: slope |13: Heart disease
            new_row = list(itemgetter(0, 3, 4, 7, 9, 10, 13)(row))
            data.append(new_row)
    return data

def update_centroids(data_set, assignments):
    """
    Accept a dataset and a list of assignments
    return new centroids based on average of points
    """
    new_means = defaultdict(list)
    new_centroids = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
        
    for points in new_means.values():
        new_centroids.append(average(points))
    return new_centroids

def assign_points(data_points, centroids):
    """
    Assign n points to n corresponding centroids which are nearest
    i.e. each point in the data_points is assigned an index that corresponds
    to the index of a centroid in the centroids list 
    """
    assignments = []
    for point in data_points:
        shortest = float("inf")  # positive infinity
        shortest_index = 0
        for i in range(len(centroids)):
            val = dist(point, centroids[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments

def generate_k():
    """
    generate k = 2 random points. The points are 6-dimensional and the ranges correspond to the ranges of numbers in the selected features 
    """
    a = [np.random.uniform(29, 78), np.random.uniform(100, 181), np.random.uniform(187, 340), \
         np.random.uniform(90, 190), np.random.uniform(0, 3.7), round(np.random.uniform(0, 3))]
    b = [np.random.uniform(29, 78), np.random.uniform(100, 181), np.random.uniform(187, 340), \
         np.random.uniform(90, 190), np.random.uniform(0, 3.7), round(np.random.uniform(0, 3))]
    return [a, b]
         
def k_means(dataset, k = 2, iteration = 50):
    k_points = generate_k()
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    i = 1
    while (assignments != old_assignments) and (i <= iteration):
        new_centroids = update_centroids(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centroids)
        i += 1
    return assignments, new_centroids 

def main():
    data = loadData()
    data = [[float(x) for x in row] for row in data]
    relevant_data = [i[:6] for i in data]
    clusters = k_means(relevant_data, 2)[0]

    ### Evaluation ###
    
    known_tags = [bool(i[6]) for i in data]
    assert (len(clusters) == len(known_tags))
    
    # number of false negatives
    false_neg = sum(1 for i in range(len(clusters)) if (bool(clusters[i]) != known_tags[i]) and (clusters[i] == 0))

    # number of false positives
    false_pos = sum(1 for i in range(len(clusters)) if (bool(clusters[i]) != known_tags[i]) and (clusters[i] == 1))
    
    # number of true positives 
    true_pos = sum(1 for i in range(len(clusters)) if (bool(clusters[i]) == known_tags[i] == True))

    # number of true negatives
    true_neg = sum(1 for i in range(len(clusters)) if (bool(clusters[i]) == known_tags[i] == False))

    # accuracy =
    #           (num of true positives + num of true negatives)
    #        ------------------------------------------------------
    # num of true positives + num of false positives + num of false negatives + num of true negatives)
    accuracy = (true_pos + true_neg) / (false_neg + false_pos + true_pos + true_neg)

    precision = (true_pos) / (true_pos + false_pos)

    recall = (true_pos) / (true_pos + false_neg)

    print("Results (for iteration <= 50)...")
    print("accuracy = ", accuracy)
    print("precision = ", precision)
    print("recall = ", recall)

    # Finding the relation between cluster errors and number of iterations #
    err = []
    for i in range(1, 51):
        index_list = k_means(relevant_data, 2, i)[0]
        cluster_1 = [i for i, x in enumerate(index_list) if x == 0]
        cluster_2 = [i for i, x in enumerate(index_list) if x == 1]
        centroids = k_means(relevant_data, 2, i)[1]
        err.extend([sum([dist(centroids[0], i) for i in cluster_1]), sum([dist(centroids[1], i) for i in cluster_2])])

    plt.xlabel('Number of iterations')
    plt.ylabel('Erros in each cluster')
    plt.plot([y for x in range(1, 51) for y in [x, x]], err)
    plt.show()
    
    
if __name__ == "__main__":
    main()
