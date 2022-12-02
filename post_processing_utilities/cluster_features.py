# Copyright (c) Northwestern Argonne Institute of Science and Engineering (NAISE)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
from pathlib import Path

import numpy as np
import torch

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn_som.som import SOM
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD as SVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from joblib import dump, load

def get_args_parser():
    parser = argparse.ArgumentParser('Features post processing using dimensionality reduction and clustering', add_help=False)
    parser.add_argument('--features_path', default='', type=str, help="Path to the features to be processed.")

    # For the dim red method
    parser.add_argument('--dimensions', default=2, type=int, help='Reduce to this number of dimensions (Default 2).')
    parser.add_argument('--dim_red_method', default='PCA', type=str, help="Dimensionality reduction method (Default: PCA).")


    # For the clustering method
    parser.add_argument('--clustering_method', default='DBSCAN', type=str, help="Clustering method (Default: DBSCAN).")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="""This is for the DBSCAN algorithm.
            The maximum distance between two samples for one to be considered as in the neighborhood of the other (default=0.5).""")
    parser.add_argument('--dbscan_min_samples', default=5, type=int, help="""This is for the DBSCAN algorithm.
            The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself (Default 5).""")

    # General arguments
    parser.add_argument('--output_dir', default=None, help='Path where to save clustering results.')
    parser.add_argument('--subsample_feats', default=None, type=int, help='Subsample the feature space to a reduce number of samples.')

    return parser








def process_features(args):
    (scale_model, x) = bring_features(args)
    (dim_red_model, x) = reduce_dim(x, args.dim_red_method, args.dimensions)
    y = cluster_data(x, args)

    clusters = {'x': x, 'y': y}

    return clusters, dim_red_model, scale_model


def bring_features(args):
    feats = torch.load(args.features_path)
    if args.subsample_feats:
        feats = choose_random_rows(feats, args.subsample_feats)

    scale = StandardScaler()
    feats = scale.fit_transform(feats)
    return scale, feats


def reduce_dim(feats, method='PCA', dimensions=2):
    if method == 'PCA':
        #pca = PCA(n_components='mle', svd_solver='full')
        #pca = PCA(n_components=0.99999, svd_solver='full')
        pca = PCA(n_components=dimensions, svd_solver='full')
        pca.fit(feats)
        return pca, pca.transform(feats)
    elif method == 'SVD':
        svd = SVD(n_components=dimensions)
        svd.fit(feats)
        return svd, svd.transform(feats)
    else:
        raise NameError(f"Unknow dimensionality reduction method: {method}")



def cluster_data(data, args):
    if args.clustering_method == 'DBSCAN':
        # Compute Density-based spatial clustering of applications with noise (DBSCAN)
        labels = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples).fit_predict(data)
    elif args.clustering_method == 'KMEANS':
        # Compute K-Means clustering 
        labels = KMeans(init="k-means++", n_clusters=args.kmeans_n_clusters, n_init=args.kmeans_n_init).fit_predict(data)
    elif args.clustering_method == 'SOM':
        # Comput SOM clustering
        labels = SOM(m=args.som_m, n=args.som_n, dim=args.som_dim).fit_predict(data)
    else:
        raise NameError(f"Unknow clustering algorithm method: {args.clustering_method}")

    return labels


def choose_random_rows(an_array, n_samples):
    number_of_rows = an_array.shape[0]
    random_indices = np.random.choice(number_of_rows, size=n_samples, replace=False)
    random_rows = an_array[random_indices, :]
    return random_rows


def save_process(args, clusters, dim_red_model, scale_model):
    # save results
    os.makedirs(args.output_dir, exist_ok=True)
    clusters_fname = os.path.join(args.output_dir, "clusters")
    np.save(clusters_fname, clusters)
    print(f"{clusters_fname} saved.")

    red_dim_model_fname = os.path.join(args.output_dir, "dim_red_model")
    dump(dim_red_model, red_dim_model_fname)
    print(f"{red_dim_model_fname} saved.")

    scale_model_fname = os.path.join(args.output_dir, "scale_model")
    dump(scale_model, scale_model_fname)
    print(f"{scale_model_fname} saved.")





if __name__ == '__main__':
    parser = argparse.ArgumentParser('Cloud-DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (clusters, dim_red_model, scale_model) = process_features(args)

    if args.output_dir:
        save_process(args, clusters, dim_red_model, scale_model)

