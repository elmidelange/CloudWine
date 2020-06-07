import logging
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# A dataset class that handles preprocessing, augementaion and saving
class Dataset:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df_raw = self.__read_data__()
        self.df = self.__process_data__()

    # Returns the dataset corpus
    def get_corpus(self):
        return self.df['description'].to_list()

    # Returns corpus labels
    def get_labels(self):
        return self.df['variety_region']

    # Returns dataset
    def get_df(self):
        return self.df

    # Saves the dataframe as a pickle
    def save(self, dir):
        columns = ['title', 'description', 'points', 'price', 'variety', 'region_1']
        self.df[columns].to_pickle(dir + '/tfidf_metadata.pkl')

    # Read in dataframe
    def __read_data__(self):
        # path = self.dir + '/sample.csv'
        # path = self.dir + '/sample_10000.csv'
        # path = self.dir + '/winemag-data-130k-v2.csv'
        path = self.filepath
        logger.info("Loading data from %s", path)
        if not (os.path.isfile(path)):
            raise ValueError(path)
        return pd.read_csv(path)

    # Process dataframe
    def __process_data__(self):
        # Remove nans from important columns
        df = self.df_raw[(self.df_raw['variety'].notna()) & (self.df_raw['region_1'].notna())]
        df = df.reset_index(drop=True)
        # Create new feature variety + region_1
        df['variety_region'] = df[['variety', 'region_1']].agg('-'.join, axis=1)
        return df


# A model validation class that evaluate model performance
class Validation:
    def __init__(self):
        self.top_labels = ['Cabernet Sauvignon-Napa Valley','Nebbiolo-Barolo',
            'Pinot Noir-Willamette Valley','Pinot Noir-Russian River Valley',
            'Champagne Blend-Champagne','Pinot Noir-Sonoma Coast','Malbec-Mendoza',
            'Chardonnay-Russian River Valley','Sangiovese-Brunello di Montalcino',
            'Nebbiolo-Barbaresco']
        self.top_n = 6

    # Returns the average cosine simialrity for top n clusters
    def cluster_similarities(self, x, df):
        logger.info("Calculating cluster similarities")
        df = df[df['variety_region'].isin(self.top_labels[:self.top_n])]
        labels = df['variety_region'].unique()
        vals = []
        for l in labels:
            idx = df[df['variety_region'] == l].index.to_numpy()
            x_cluster = x[idx]
            cluster_mean = np.mean(x_cluster, axis=0)
            out = np.mean(cosine_similarity([cluster_mean], x_cluster).flatten())
            vals += [{l: out}]
        return vals

    # Plots the document embeddings in 2D using PCA
    def plot_pca(self, x,y, top=True):
        logger.info("Running PCA analysis")
        feat_cols = [ 'embedding'+str(i) for i in range(x.shape[1]) ]
        df_embed = pd.DataFrame(x,columns=feat_cols)
        df_embed['y'] = y

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(x)
        df_embed['pca-one'] = pca_result[:,0]
        df_embed['pca-two'] = pca_result[:,1]

        if top==True:
            # df_embed.loc[~df_embed['y'].isin(self.top_labels[:self.top_n]), 'y'] = 'other'
            # df_embed['size'] = 0
            # df_embed.loc[df_embed['y'] != 'other','size'] = 10
            df_embed = df_embed[df_embed['y'].isin(self.top_labels[:self.top_n])]

        plt.figure(figsize=(14,8))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            # size="size",
            sizes=(100,400),
            data=df_embed,
            legend="full",
            alpha=0.9
        )
        plt.title('TF-IDF Text Embeddings (2D PCA) - ' + str(len(df_embed)) + ' samples')
        plt.show()
