---
layout: post
title: Blog Post 2 - Spectral Clustering 
---

This blog post will explain a simple version of the clustering algorithm for clustering data points.

## Part A

This blogpost will be using multiple modules to graph the data accordingly, so let's import those modules right now!

```python
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
import sklearn
```

In general, here's the usage for each module in this clustering algorithm:
- `numpy` will be used for manipulating arrays in a convenient way
- `sklearn` will be used for analyzing matrices 
- `matplotlib` will be used to graph the data points

Let's first create the dataset that we want to color-code based on cluster. Let's name the dataset `X` and have it contain 200 data points: 

```python
np.random.seed(1234)
n = 200
X, y = datasets.make_moons(n_samples=n, shuffle=True, noise=0.05, random_state=None)
plt.scatter(X[:,0], X[:,1])
```
![hw2_blue_cluster.png]({{ site.baseurl }}/images/hw2_blue_cluster.png)

Notice how there are clearly two different moon-shaped clusters on the graph! Because of this irregular shape, we'll need to execute a simple clustering algorithm.

For Part A, we first have to create the similarity matrix **A**. which is a 2D `numpy` array with shape `(n,n)` where `n` is the number of data points. In our example above, we have `n = 200`. 

To construct this matrix, we will use a positive parameter `epsilon` where `A[i,j] = 1` if `X[i]` is within distance `epsilon` of `X[j]`. If not, then `A[i,j] = 0`. In addition, the diagonal entries `A[i,i] = 0` as well. To execute this, we will do the following: 

1. Create a matrix containing all the distances between points. Luckily, `sklearn` has a handy function that does just that! So, we'll use `sklearn.metrics.pairwise_distances()`, which will result in an `nxn` matrix.
2.  We will then need to create boolean masks to denote which values should be `0` and which ones should be `1`. 
3. We will manually fill the diagonal with `0`s using `np.fill_diagonal()` since its distances will be `0` which is less than `epsilon`, so it would result in being `1` instead of the correct `0`. 

In this example, let `epsilon = 0.4`. All together, the code should look like so: 

```python
epsilon = 0.4

#create a matrix containing all the distances (size nxn because X is nxn)
A = sklearn.metrics.pairwise_distances(X)

#create a boolean mask -- if the distance in (i,j) < epsilon, make the (i,j) entry = 1
mask_close = A < epsilon

#if not within epsilon distance, make (i,j) entry = 0
mask_far = A >= epsilon
A[mask_close] = 1
A[mask_far] = 0

#fill the diagonal with 0s
np.fill_diagonal(A,0)

#see what A looks like
print(A)
```
```
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]
 ...
 [0. 0. 0. ... 0. 1. 1.]
 [0. 0. 1. ... 1. 0. 1.]
 [0. 0. 0. ... 1. 1. 0.]]
```
