import os
import glob
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go
import logging
import gdown

from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass

data_file = 'cloudwine/data/data.csv'
model_dir = 'cloudwine/models/'


# Logging
logger = logging.getLogger()
handler = logging.StreamHandler()
formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.ERROR)


@dataclass
class Embeddings:
    def __init__(self, dir):
        self.bert_model = pickle.load(open(dir + "bert_model.pkl", 'rb'))
        self.bert_vectors = pickle.load(open(dir + "bert_vectors.pkl", 'rb'))
        self.docvec_model = pickle.load(open(dir + "doc2vec_model.pkl", 'rb'))
        self.docvec_vectors = pickle.load(open(dir + "doc2vec_vectors.pkl", 'rb'))
        self.tfidf_model = pickle.load(open(dir + "tfidf_model.pkl", 'rb'))
        self.tfidf_vectors = pickle.load(open(dir + "tfidf_vectors.pkl", 'rb'))


@dataclass
class DataModule:
    data: pd.DataFrame
    data_filtered: pd.DataFrame
    model: None
    vectors: np.ndarray


# Download data from GCP bucket
@st.cache(suppress_st_warning=True)
def download_data():
    logger.info('\n\n DOWNLOADING DATA \n\n')
    app_path = os.path.abspath(os.getcwd())
    files = glob.glob1(app_path + '/cloudwine/models/',"*.pkl")
    pklCounter = len(files)
    print(f'Found {pklCounter} models')
    if pklCounter != 6:
        print('Removing items')
        gd_links = {'tfidf_vectors.pkl':'https://drive.google.com/uc?export=download&id=1lsgOd_onTor8_5MW4y-Za96B5wCPoVO8',
            'tfidf_model.pkl': 'https://drive.google.com/uc?export=download&id=1BduFhJHYS0YOmgUaXOz2t7F9u4kuxW4w',
            'doc2vec_vectors.pkl': 'https://drive.google.com/uc?export=download&id=1hW75zEJFCza55TiSVk54bUhXMnddrCRJ',
            'doc2vec_model.pkl': 'https://drive.google.com/uc?export=download&id=15wd7tgNvpTKbIdNowle3gLLedcInOUM6',
            'bert_vectors.pkl': 'https://drive.google.com/uc?export=download&id=1edPl0phsC7N2PfWpUb2VLS6NitjIBFfS',
            'bert_model.pkl': 'https://drive.google.com/uc?export=download&id=1osi-h2TgCVbKZWfHmdnI5mKQO6GWi0Tb'
        }
        for name, url in gd_links.items():
            print(name, url)
            gdown.download(url, f'{app_path}/cloudwine/models/{name}', quiet=False)

class Loader:
    def __init__(self, title=None):
        if title:
            st.write(title)
        self.prog_bar = st.progress(0)

    def update_loader(self, progress):
        self.prog_bar.progress(progress)

    def remove_loader(self):
        self.prog_bar.empty()

# Initialise data class
def init_data():
    global embeddings
    app_path = os.path.abspath(os.getcwd())
    model_dir = app_path + '/cloudwine/models/'
    embeddings = Embeddings(model_dir)
    model = pickle.load(open(model_dir + "bert_model.pkl", 'rb'))
    vectors = pickle.load(open(model_dir + "bert_vectors.pkl", 'rb'))
    df = pd.read_csv(data_file)
    return DataModule(df, df, model, vectors)


# Update the embedding type
def update_embedding(data_module, embed_model):
    if embed_model == "BERT":
        data_module.model = embeddings.bert_model
        data_module.vectors = embeddings.bert_vectors
    elif embed_model == "Doc2Vec":
        data_module.model = embeddings.docvec_model
        data_module.vectors = embeddings.docvec_vectors
    if embed_model == "TF-IDF":
        data_module.model = embeddings.tfidf_model
        data_module.vectors = embeddings.tfidf_vectors


# Show grouped bar graph for model evaluation metrics
def show_metrics_graph():
    app_path = os.path.abspath(os.getcwd())
    df = pickle.load(open(app_path + "/cloudwine/resources/graph_data.pkl", 'rb'))

    models=['tfidf', 'doc2vec', 'bert']
    fig = go.Figure(data=[
        go.Bar(name='No Preprocessing', x=models, y=df[df['processed']==0]['similarity'], text=df[df['processed']==0]['similarity'].round(2),
            textposition='auto',),
        go.Bar(name='Text Preprocessing', x=models, y=df[df['processed']==1]['similarity'], text=df[df['processed']==1]['similarity'].round(2),
            textposition='auto',)
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', xaxis_title='Embedding Type', yaxis_title="Intra-cluster Cosine Similarity")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
