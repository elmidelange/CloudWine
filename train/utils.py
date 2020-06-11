import logging
import os
import pickle

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

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
    def __init__(self, filepath, args):
        self.filepath = filepath
        self.df_raw = self.__read_data__()
        self.df_processed = self.__process_data__()
        self.df = self.__transform_data__(lowercase=args['lowercase'],
            punctuation=args['remove_punctuation'], stop_words=args['remove_stopwords'],
            lemmatize=args['lemmatize']
        )

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
    def save(self, path):
        path = path.replace('raw', 'processed')
        columns = ['title', 'description', 'points', 'price', 'variety', 'region_1', 'variety_region']
        print('Saving to ' + path)
        self.df[columns].to_csv(path, index=False)

    # Read in dataframe
    def __read_data__(self):
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

    # Transform dataframe
    def __transform_data__(self, lowercase=True, punctuation=True, stop_words=True, lemmatize=True):
        corpus = self.df_processed['description']
        if lowercase:
            # Convert text to lowercase
            logger.info('Converting to lowercase')
            corpus = corpus.str.lower()
        if punctuation:
            # Remove punctuation
            logger.info('Removing punctuation')
            corpus = corpus.str.replace('[^\w\s]','')
        if stop_words:
            # Remove stopwords
            logger.info('Removing stopwords')
            def remove_stops(text):
                stops = stopwords.words('english')
                word_list = text.split()
                meaningful_words = [w for w in word_list if not w in stops]
                return ' '.join(meaningful_words)
            corpus = corpus.apply(remove_stops)
        if lemmatize:
            # Lemmatize words
            logger.info('Lemmatizing')
            def lemmatize_text(text):
                # w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
                lemmatizer = nltk.stem.WordNetLemmatizer()
                word_list = text.split()
                lemmatized_words = [lemmatizer.lemmatize(w) for w in word_list]
                return ' '.join(lemmatized_words)
            corpus = corpus.apply(lemmatize_text)
        df = self.df_processed.copy()
        df['description'] = corpus
        return df


# A model validation class that evaluate model performance
class Validation:
    def __init__(self):
        self.top_labels = ['Cabernet Sauvignon-Napa Valley','Nebbiolo-Barolo',
            'Pinot Noir-Willamette Valley','Pinot Noir-Russian River Valley',
            'Champagne Blend-Champagne','Pinot Noir-Sonoma Coast','Malbec-Mendoza',
            'Chardonnay-Russian River Valley','Sangiovese-Brunello di Montalcino',
            'Nebbiolo-Barbaresco']
        self.top_n = 10

    # Returns the average cosine simialrity for top n clusters
    def cluster_similarities(self, x, df):
        logger.info("Calculating cluster similarities")
        df = df[df['variety_region'].isin(self.top_labels[:self.top_n])]
        labels = df['variety_region'].unique()
        vals = []
        similarity_sum = 0
        count = 0
        for l in labels:
            idx = df[df['variety_region'] == l].index.to_numpy()
            x_cluster = x[idx]
            cluster_mean = np.mean(x_cluster, axis=0)
            out = np.mean(cosine_similarity([cluster_mean], x_cluster).flatten())
            similarity_sum += out
            count += 1
            vals += [{l: out}]
        return {'similarity':similarity_sum/count, 'detail': vals}

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
