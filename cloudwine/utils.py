import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


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

# Show t-SNE 2D plot of embeddings
def show_tsne(model, data_file, model_dir):
    # Load wine reviews dataset
    df = pd.read_csv(data_file)

    df_tsne = pickle.load(open(model_dir + "tsne_2d.pkl", 'rb'))

    # Load trained embeddings
    if model == 'TF-IDF':
        tsne_result = df_tsne[['tfidf_1', 'tfidf_2']].values
        x = pickle.load(open(model_dir + "tfidf_vectors.pkl", 'rb'))
    elif model == 'Doc2Vec':
        tsne_result = df_tsne[['doc2vec_1', 'doc2vec_2']].values
        x = pickle.load(open(model_dir + "doc2vec_vectors.pkl", 'rb'))
    elif model == 'BERT':
        tsne_result = df_tsne[['bert_1', 'bert_2']].values
        x = pickle.load(open(model_dir + "bert_vectors.pkl", 'rb'))

    # Graph
    df_graph = df[['title', 'variety_region']]
    df_graph['tsne-one'] = tsne_result[:,0]
    df_graph['tsne-two'] = tsne_result[:,1]

    top_label = 'Cabernet Sauvignon-Napa Valley'

    df_cab = df_graph[df_graph['variety_region'] == top_label]

    st.write(df_cab)

    df_cab['tsne-one'] = df_cab['tsne-one'] - df_cab['tsne-one'].mean()
    df_cab['tsne-two'] = df_cab['tsne-two'] - df_cab['tsne-two'].mean()

    st.write(df_cab)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_cab['tsne-one'],
        y=df_cab['tsne-two'],
        name="Other",
        mode="markers",
        marker=go.scatter.Marker(
            # size=sz,
            # color=df_background['label'],
            opacity=0.6,
            colorscale="Viridis"
        )
    ))

    fig.update_xaxes(range=[-0.01, 0.01])
    fig.update_yaxes(range=[-0.01, 0.01])

    st.plotly_chart(fig, use_container_width=True)


# Show PCA 2D plot of embeddings
def show_pca(model, data_file, model_dir):
    # Load wine reviews dataset
    df = pd.read_csv(data_file)

    df_tsne = pickle.load(open(model_dir + "pca_2d.pkl", 'rb'))

    # Load trained embeddings
    if model == 'TF-IDF':
        tsne_result = df_tsne[['tfidf_1', 'tfidf_2']].values
        x = pickle.load(open(model_dir + "tfidf_vectors.pkl", 'rb'))
    elif model == 'Doc2Vec':
        tsne_result = df_tsne[['doc2vec_1', 'doc2vec_2']].values
        x = pickle.load(open(model_dir + "doc2vec_vectors.pkl", 'rb'))
    elif model == 'BERT':
        tsne_result = df_tsne[['bert_1', 'bert_2']].values
        x = pickle.load(open(model_dir + "bert_vectors.pkl", 'rb'))

    # Graph
    df_graph = df[['title', 'variety_region']]
    df_graph['tsne-one'] = tsne_result[:,0]
    df_graph['tsne-two'] = tsne_result[:,1]

    top_label = 'Cabernet Sauvignon-Napa Valley'

    df_cab = df_graph[df_graph['variety_region'] == top_label]

    st.write(df_cab)

    df_cab['tsne-one'] = df_cab['tsne-one'] - df_cab['tsne-one'].mean()
    df_cab['tsne-two'] = df_cab['tsne-two'] - df_cab['tsne-two'].mean()

    st.write(df_cab)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_cab['tsne-one'],
        y=df_cab['tsne-two'],
        name="Other",
        mode="markers",
        marker=go.scatter.Marker(
            # size=sz,
            # color=df_background['label'],
            opacity=0.6,
            colorscale="Viridis"
        )
    ))

    fig.update_xaxes(range=[-1, 1])
    fig.update_yaxes(range=[-1, 1])

    st.plotly_chart(fig, use_container_width=True)

    # Cosine similarity with mean for top 5 labels
    top_labels = ['Cabernet Sauvignon-Napa Valley','Nebbiolo-Barolo',
        'Pinot Noir-Willamette Valley','Pinot Noir-Russian River Valley',
        'Champagne Blend-Champagne','Pinot Noir-Sonoma Coast','Malbec-Mendoza',
        'Chardonnay-Russian River Valley','Sangiovese-Brunello di Montalcino',
        'Nebbiolo-Barbaresco']

    df_subset = df[df['variety_region'].isin(top_labels)]
    x_subset = x[df_subset.index.to_numpy()]

    labels = df_subset['variety_region'].unique()
    vals = []
    for l in labels:
        # Grab all the vectors for label l
        idx = df_subset[df_subset['variety_region'] == l].reset_index().index.to_numpy()
        x_cluster = x_subset[idx]
        # Calculate the mean vector
        cluster_mean = np.mean(x_cluster, axis=0)
        # Perform cosine similarity with mean
        out = float(np.mean(cosine_similarity([cluster_mean], x_cluster).flatten()))
        vals += [{l: out}]

    st.write(vals)
