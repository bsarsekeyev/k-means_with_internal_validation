K-Means
=======

## General description
 
Python implementation of k-means clustering algorithm with Davies-Bouldin internal cluster validation and Minmax initialization methods.

Internal validation is helful to determine and test the number of centroids.
Internal cluster validation evaluates the goodness of a clustering structure and estimates the number of appropriate clusters without reference to external data (such as labels).

Minmax selects an arbitrary point as the first centroid and then adds new centroids one by one. At each step, the next centroid is the point that is furthest (max) from its nearest (min) existing centroid.

## Input

Iris-bezdek dataset.

## Output
SSE loss and Davies-Bouldin Index

