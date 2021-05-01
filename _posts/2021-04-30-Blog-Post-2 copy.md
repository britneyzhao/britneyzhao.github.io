---
layout: post
title: Blog Post 2 - Spectral Clustering 
---

This blog post will explain a simple version of the clustering algorithm for clustering data points.

## Part A

First, let us gather the data about the Palmer Penguins into a dataset that we can manipulate using Python. To do so, we will use the `pandas` module. The function `read_csv` from `pandas` reads a csv file into a `pandas` DataFrame. This will allow us to gather the data so that we can display certain parts of it. This function can work using a website URL that contains the data you want. The following code executes this by reading the Palmer Penguin data from a website containing it: 

```python
import pandas as pd
url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/palmer_penguins.csv"
penguins = pd.read_csv(url)
```
