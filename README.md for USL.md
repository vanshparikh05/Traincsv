
# K-Means Clustering and PCA from Scratch

This project demonstrates the implementation of two fundamental machine learning techniques — **K-Means Clustering** and **Principal Component Analysis (PCA)** — entirely from scratch using only basic Python libraries: NumPy, Pandas, and Matplotlib.

## Dataset Used
The Iris dataset is used, which consists of 150 samples from three species of Iris flowers. Each sample has four features: sepal length, sepal width, petal length, and petal width.

## K-Means Clustering

K-Means is an unsupervised learning algorithm used to group data into `k` clusters. The steps followed in the implementation are:

1. Randomly select `k` initial cluster centers (centroids) from the data.
2. For each data point, compute its distance to each centroid and assign it to the nearest one.
3. Recalculate the centroids by computing the mean of all points assigned to each cluster.
4. Repeat the assignment and update steps until the centroids no longer change significantly.

This results in a set of clusters where each point belongs to the cluster with the nearest mean.

## Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that projects data into a lower-dimensional space while preserving as much variance as possible. The steps followed in the implementation are:

1. Center the data by subtracting the mean of each feature.
2. Compute the covariance matrix of the centered data.
3. Calculate the eigenvalues and eigenvectors of the covariance matrix.
4. Sort the eigenvectors by descending eigenvalues to select the principal components.
5. Project the data onto the selected principal components to obtain the reduced representation.

In this project, PCA reduces the dataset from 4 dimensions to 2, allowing for visualization.

## Visualization

The final output visualizes the 2D PCA projection of the dataset, with each point colored according to the cluster assigned by the K-Means algorithm. This helps understand how well the clustering algorithm has separated the underlying classes.

## Objective

This project is purely educational, aiming to provide a foundational understanding of K-Means and PCA by implementing them manually rather than relying on libraries like Scikit-learn.
