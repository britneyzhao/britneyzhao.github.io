---
layout: post
title: Blog Post 2 - Spectral Clustering 
---

This blog post will explain a simple version of the clustering algorithm for clustering data points.

## Part A - Similarity Matrix

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

Our matrix looks great! Filled with only 0s and 1s. 

## Part B - Binary Norm Cut Objective

Now that **A** contains information about which points are near which points, we will now try to cluster the data points accordingly. First, let's define some mathematical expressions: 

Let $$d_i = \sum_{j = 1}^n a_{ij}$$ be the $$i$$th row-sum of $$\mathbf{A}$$, which is also called the *degree* of $$i$$. Let $$C_0$$ and $$C_1$$ be two clusters of the data points. We assume that every data point is in either $$C_0$$ or $$C_1$$. The cluster membership as being specified by `y`. We think of `y[i]` as being the label of point `i`. So, if `y[i] = 1`, then point `i` (and therefore row $$i$$ of $$\mathbf{A}$$) is an element of cluster $$C_1$$.  

Now, to cluster the data points, we will compute the *binary norm cut objective* of the matrix **A**, which is the following: 

$$N_{\mathbf{A}}(C_0, C_1)\equiv \mathbf{cut}(C_0, C_1)\left(\frac{1}{\mathbf{vol}(C_0)} + \frac{1}{\mathbf{vol}(C_1)}\right)\;.$$

In this expression, 
- $$\mathbf{cut}(C_0, C_1) \equiv \sum_{i \in C_0, j \in C_1} a_{ij}$$ is the *cut* of the clusters $$C_0$$ and $$C_1$$. 
- $$\mathbf{vol}(C_0) \equiv \sum_{i \in C_0}d_i$$, where $$d_i = \sum_{j = 1}^n a_{ij}$$ is the *degree* of row $$i$$ (the total number of all other rows related to row $$i$$ through $$A$$). The *volume* of cluster $$C_0$$ is a measure of the size of the cluster. 

Let's first focus on the cut term. Based on the formula, we can compute this term by summing up the entries `A[i,j]` for each pair of points `(i,j)` in different cluster. Since `y[i]` signifies which cluster the data point is in, we can write the code for a `cut()` function, which takes in the similarity matrix `A` and the cluster membership `y` and outputs the value of the cut term, like so:

```python
"""
@param A: similarity matrix
@param y: cluster membership vector
@return the cut term
"""
def cut(A,y):
    length, width = A.shape
    cut = 0
    
    #loop through each element in A and sum up the points in different clusters as defined above
    for i in range(length):
        for j in range(width):
            if y[i] != y[j]:
                cut += A[i][j]
    
    return cut
```

Let's test this out with our similarity matrix **A** and cluster membership vector `y`. To compare, the cut object should be much smaller to a randomally generated vector. So, let's compare these two values like so: 

```python
cut_real = cut(A,y) #13.0
rand_v = np.random.randint(0,2, size = (n,1)) 
cut_rand = cut(A,rand_v)

print(cut_real)
print(cut_rand)
```
```
13.0
2242.0
```

Our correct cut object is much smaller, which means we're on the right track!

Next, let's compute the volume term. To do so, we want the sum of row `i` if `y[i] == 0` for $$C_0$$ and `y[i] == 1` for $$C_1$$. We can compute this very easily and efficiently using list comprehension! So, let's create a `vols()` function that returns the volumes of each cluster as a tuple:

```python
"""
@param A: similarity matrix
@param y: cluster membership vector
@return the volumes of each cluster as a tuple
"""
def vols(A,y):
    length, width = A.shape
    
    vol0 = sum([sum(A[i]) for i in range(length) if y[i] == 0])
    vol1 = sum([sum(A[i]) for i in range(length) if y[i] == 1])
    
    return (vol0, vol1)
```

Now that we have both `vols()` and `cut()` defined, we can create a `normcut()` function that uses these two functions to compute the normcut objective! Based on the formula, `normcut()` will be like so:

```python
def normcut(A,y):
    v0, v1 = vols(A,y)
    return cut(A,y)*((1/v0) + (1/v1))
```

Now, let's test this function! To compare, the norm cut of the correct labels should be significantly smaller than the norm cut of a randomly generated vector: 

```python
print(normcut(A,y))
print(normcut(A, rand_v))
```
```
0.02303682466323045
1.9858316719148859
```

Notice that 0.023 is a lot smaller than 1.986, so we're on the right track!

## Part C - Another Approach at Norm Cut

Our approach works perfectly, however, using linear algebra, we can also derive the following vector **z** such that: 

$$
z_i = 
\begin{cases}
    \frac{1}{\mathbf{vol}(C_0)} &\quad \text{if } y_i = 0 \\ 
    -\frac{1}{\mathbf{vol}(C_1)} &\quad \text{if } y_i = 1 \\ 
\end{cases}
$$

Using this vector **z**, we can compute the norm cut like so: 

$$\mathbf{N}_{\mathbf{A}}(C_0, C_1) = 2\frac{\mathbf{z}^T (\mathbf{D} - \mathbf{A})\mathbf{z}}{\mathbf{z}^T\mathbf{D}\mathbf{z}}\;,$$

where $$\mathbf{D}$$ is the diagonal matrix with nonzero entries $$d_{ii} = d_i$$, and  where $$d_i = \sum_{j = 1}^n a_i$$ is the degree (row-sum) from before.

Note that `z[i]` will be greater than 0 if the point is in $$C_0$$ and will be less than 0 if the point is in $$C_1$$. With this new approach, let's create a new function `transform()` which will return the appropriate **z** vector based on the formula above:

```python
#create z based on the formula
def transform(A,y):
    #get the volumes of the clusters
    v0, v1 = vols(A,y)
    #first fill z with 1/v0
    z = np.array([1/v0 for i in range(len(y))])
    
    #create a boolean mask to see which values should be -1/v1
    mask = y == 1
    z[mask] = -1/v1
    
    return z
```

To test our function, let's compute the matrix product and see how it compares to our original norm cut value from Part B. Let's also check that $$\mathbf{z}^T\mathbf{D}\mathbb{1} = 0$$, where $$\mathbb{1}$$ is the vector of `n` ones, which should be true if our **z** is correct. We'll use `np.isclose()` to determine if the two norm cut calculations are about equal, because rounding could result in `False`.

```python
#create vector z and its transpose
z = transform(A,y)
z_T = z.T

#create D by first making a nxn array with the diagonal entries as d_i
D = np.array([[sum(A[i]) if i == j else 0 for i in range(len(y))] for j in range(len(y))])

#based on the matrix multiplication formula
norm1 = 2*(z_T@(D-A)@z)/(z_T@D@z)
#from Part B
norm2 = normcut(A,y)

print(np.isclose(norm1, norm2))

print(z_T@D@(np.ones(n)))
```
```
True
0.0
```

Looks like everything checks out!


## Part D - Minimizing a Function

The matrix multiplication above can be optimized by substituing **z** for its orthogonal complement relative to $$\mathbf{D}\mathbb{1}$$ and optimizing that. 

The following code define the orthogonal complement calculation:

```python
def orth(u, v):
    return (u @ v) / (v @ v) * v

e = np.ones(n) 

d = D @ e

def orth_obj(z):
    z_o = z - orth(z, d)
    return (z_o @ (D - A) @ z_o)/(z_o @ D @ z_o)
```

For this part, let's use the `minimize` function from `scipy.optimize` to minimize this function `orth_obj()` like so:

```python
import scipy
z_ = scipy.optimize.minimize(orth_obj, z)
```

Now, `z_` is our minimized **z** vector. 

## Part E - Graphing Clusters with the Minimized Vector

Let's plot the moon-shaped clusters again using this optimized `z_` vector to dictate which color each data point is. In theory, `z_[i]` should be greater than 0 when it is in $$C_0$$, but since this isn't the best optimization problem, we'll make the bottom threshold a small negative number to accommodate for this error. In this instance, I chose `-0.0015`:

```python
plt.scatter(X[:,0], X[:,1], c = [z_.x < -0.0015])
plt.show()
```
![hw2_firstcolorplot.png]({{ site.baseurl }}/images/hw2_firstcolorplot.png)

Notice how we came close, but it isn't perfect! The left tip of the bottom moon-shaped cluster is purple instead of yellow... So let's make the optimization a little better!

## Part F - Laplacian Matrix


