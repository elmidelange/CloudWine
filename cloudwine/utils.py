import os
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go

from sklearn.metrics.pairwise import cosine_similarity
from google.cloud import storage
from dataclasses import dataclass

data_file = 'cloudwine/data/data.csv'
model_dir = 'cloudwine/models/'

@dataclass
class Embeddings:
    bert_model = pickle.load(open("cloudwine/models/bert_model.pkl", 'rb'))
    bert_vectors = pickle.load(open("cloudwine/models/bert_vectors.pkl", 'rb'))
    docvec_model = pickle.load(open("cloudwine/models/doc2vec_model.pkl", 'rb'))
    docvec_vectors = pickle.load(open("cloudwine/models/doc2vec_vectors.pkl", 'rb'))
    tfidf_model = pickle.load(open("cloudwine/models/tfidf_model.pkl", 'rb'))
    tfidf_vectors = pickle.load(open("cloudwine/models/tfidf_vectors.pkl", 'rb'))
embeddings = Embeddings()


@dataclass
class DataModule:
    data: pd.DataFrame
    data_filtered: pd.DataFrame
    model: None
    vectors: np.ndarray


# Download data from GCP bucket
@st.cache
def download_data():
    bucket_name = 'cloudwine'
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # Download reviews csv
    blobs = bucket.list_blobs(prefix='data/data.csv')  # Get list of files
    for blob in blobs:
        filename = blob.name.split('/')[1]
        if filename and not os.path.isfile('cloudwine/data/' + filename):
            blob.download_to_filename('cloudwine/data/' + filename)  # Download

    # Download trained embeddings
    blobs = bucket.list_blobs(prefix='model/')  # Get list of files
    for blob in blobs:
        filename = blob.name.split('/')[1]
        if filename and not os.path.isfile('cloudwine/models/' + filename):
            blob.download_to_filename('cloudwine/models/' + filename)  # Download

    # Download trained embeddings
    blobs = bucket.list_blobs(prefix='resources/')  # Get list of files
    for blob in blobs:
        filename = blob.name.split('/')[1]
        if filename and not os.path.isfile('cloudwine/resources/' + filename):
            blob.download_to_filename('cloudwine/resources/' + filename)  # Download


# Initialise data class
def init_data():
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
    df = pickle.load(open("cloudwine/resources/graph_data.pkl", 'rb'))

    models=['tfidf', 'doc2vec', 'bert']
    fig = go.Figure(data=[
        go.Bar(name='No Preprocessing', x=models, y=df[df['processed']==0]['similarity']),
        go.Bar(name='Text Preprocessing', x=models, y=df[df['processed']==1]['similarity'])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group', xaxis_title='Embedding Type', yaxis_title="Region-Variety Similarity")
    st.plotly_chart(fig, use_container_width=True)
