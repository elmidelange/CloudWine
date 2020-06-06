import os
import argparse
import pickle
import logging

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Main function
def main(args):
    # Read training data
    dataset = Dataset(args['data_dir'])
    corpus = dataset.get_corpus()

    # Train TF-IDF model
    model = TfidfTrainer()
    model.train(corpus)

    # Save run
    model.save(args['model_dir'])
    dataset.save(args['model_dir'])

    # Validate
    valid = Validation()
    x = model.get_vectors()
    df = dataset.get_df()
    valid.plot_pca(x, df['variety_region'])
    print(valid.cluster_similarities(x, df))


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


class TfidfTrainer:
    def __init__(self):
        self.tf_transformer = None
        self.text_vectors = None

    # Train TF-IDF model
    def train(self, corpus):
        logger.info("Training model")
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True, max_features = 50000)
        self.tf_transformer = tf.fit(corpus)
        self.text_vectors = tf.fit_transform(corpus).toarray()

    # Returns the trained sentence vectors
    def get_vectors(self):
        return self.text_vectors

    # Save the model in pickle format
    def save(self, path):
        logger.info("Saving model to %s", path)
        # Save TfidfVectoriser vocab
        with open(path + '/tfidf_transform.pkl', "wb") as pickleFile:
            pickle.dump(self.tf_transformer, pickleFile)
        # Save corpus text vectors
        logger.info("Saving vectors to %s", path)
        with open(path +  '/tfidf_matrix.pkl', "wb") as pickleFile:
            pickle.dump(self.text_vectors, pickleFile)


class Dataset:
    def __init__(self, dir):
        self.dir = dir
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
        path = self.dir + '/sample.csv'
        # path = self.dir + '/sample_10000.csv'
        # path = self.dir + '/winemag-data-130k-v2.csv'
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


# Checks for valid directory
def check_dir_exists(path):
    if (os.path.isdir(path)):
        return path
    else:
        raise ValueError(path)


# Returns argument parser
def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Train a TD-IDF model"
    )
    parser.add_argument(
        "-d", "--data_dir", help="Training data directory",
            default='./data/raw', type=check_dir_exists
    )
    parser.add_argument(
        "-m", "--model_dir", help="Model directory to save into",
            default='./model', type=check_dir_exists
    )
    return parser


if __name__ == "__main__":
    # execute only if run as a script
    parser = init_argparse()
    args = parser.parse_args()
    main(vars(args))
