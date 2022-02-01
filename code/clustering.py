import matplotlib.pyplot as plt
import sklearn.cluster as cluster
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist

from yellowbrick.cluster import KElbowVisualizer

import os

DATA_FOLDER = "../data"
FILE_NAME = "wholesale_customers.csv"
PLOT_FOLDER = "../plots"

def end_line():
    print(
        "------------------------------------------------------------------------------------------------------------")


# Function to explore data and understand it
# Question 2.1
def understand_data():
    # using pandas to read csv as dataframe
    df = pd.read_csv(os.path.join(DATA_FOLDER, FILE_NAME)).drop(['Channel', 'Region'], axis=1)
    metrics = df.describe()
    print("The mean of each of these values is")
    print(metrics.mean())
    print("The min of each of these values is")
    print(metrics.min())
    print("The max of each of these values is")
    print(metrics.max())
    end_line()
    return df

# Implement K Means for k=3 for 15 different combinations (6 attributes, 2 at a time)
# Question 2.2
def k_means_combos(df, K):
    km = cluster.KMeans(n_clusters=K)
    print(df.columns)
    combo = list(combinations(df.columns, 2))
    for num, c in enumerate(combo):
        X = df[[c[0], c[1]]]
        X = np.array(X)
        plt.figure()
        label = km.fit_predict(X)
        for i in range(K):
            plt.scatter(X[label == i, 0], X[label == i, 1], label=i)
        plt.legend()
        plt.xlabel(c[0])
        plt.ylabel(c[1])
        plt.title("Scatter Plot Combination %d" % (num + 1))
        plt.savefig(os.path.join(PLOT_FOLDER, 'scatter_plot_%d.png' % (num+1)))



def within_cluster_verfication(dist):
    return(dist.min(axis = 1).sum())


# Experiment to see BC and WC distances
# Question 2.3
def k_means_experiment(df, kset):
    for k in kset:
        km = cluster.KMeans(n_clusters=k)
        km.fit_predict(df)
        WC = km.inertia_
        dist = km.transform(df)**2
        # Using Scipy to calculate euclidean distance between cluster centers
        BC = (pdist(km.cluster_centers_, 'euclid')**2).sum()
        visualizer = KElbowVisualizer(km, k=(1, 20))
        visualizer.fit(df)
        visualizer.show(outpath=os.path.join(PLOT_FOLDER, 'elbow_method.png' ))
        print("K value is %d\nThe BC is %f\nThe WC (inertia) is %f\nThe BC/WC value is %f" % (k, BC, WC, BC/WC))
        print("The algorithmic calculation of WC is %d" % within_cluster_verfication(dist))
        end_line()








def main():
    df = understand_data()
    k_means_combos(df, 3)
    k_means_experiment(df, [3, 5, 10])


if __name__ == "__main__":
    main()
