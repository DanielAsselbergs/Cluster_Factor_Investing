"""
Module: clustering.py
Description: Contains functions to perform PCA, clustering (KMeans and Agglomerative),
             and evaluation of clustering quality.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score



