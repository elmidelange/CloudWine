import pandas as pd
import numpy as np
import pickle
import streamlit as st
# import plotly.figure_factory as ff
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def show_pca(model, data_file, model_dir):
    print('show_pca')
    # Load wine reviews dataset
    df = pd.read_csv(data_file)
    # Load trained embeddings
    if model == 'TF-IDF':
        x = pickle.load(open(model_dir + "tfidf_vectors.pkl", 'rb'))
    elif model == 'Doc2Vec':
        x = pickle.load(open(model_dir + "doc2vec_vectors.pkl", 'rb'))
    elif model == 'BERT':
        x = pickle.load(open(model_dir + "bert_vectors.pkl", 'rb'))
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(x)
    # Graph
    df_graph = df[['title', 'variety_region']]
    df_graph['pca-one'] = pca_result[:,0]
    df_graph['pca-two'] = pca_result[:,1]

    # Select top 5 variety_region labels
    top_labels = ['Cabernet Sauvignon-Napa Valley','Nebbiolo-Barolo',
        'Pinot Noir-Willamette Valley','Pinot Noir-Russian River Valley',
        'Champagne Blend-Champagne','Pinot Noir-Sonoma Coast','Malbec-Mendoza',
        'Chardonnay-Russian River Valley','Sangiovese-Brunello di Montalcino',
        'Nebbiolo-Barbaresco']
    # df_subset = df[df['variety_region'].isin(top_labels)]
    # x_subset = x[df_subset.index.to_numpy()]

    # df_graph['label'] = df['variety_region'].rank(method='dense', ascending=False).astype(int)
    # label =
    # df_graph['label'] = df['title'].apply(lambda x: '#FF0000' if x in top_labels else '#888888')

    top_label = 'Cabernet Sauvignon-Napa Valley'
    df_background = df_graph[df_graph['variety_region'] != top_label]
    df_foreground = df_graph[df_graph['variety_region'] == top_label]

    df_background['label'] = '#888888'
    df_foreground['label'] = '#FF0000'

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_background['pca-one'],
        y=df_background['pca-two'],
        name="Other",
        mode="markers",
        marker=go.scatter.Marker(
            # size=sz,
            color=df_background['label'],
            opacity=0.6,
            colorscale="Viridis"
        )
    ))

    fig.add_trace(go.Scatter(
        x=df_foreground['pca-one'],
        y=df_foreground['pca-two'],
        name=top_label,
        mode="markers",
        marker=go.scatter.Marker(
            # size=sz,
            color=df_foreground['label'],
            opacity=0.6,
            colorscale="Viridis"
        )
    ))

    st.plotly_chart(fig, use_container_width=True)

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
        idx = df_subset[df_subset['variety_region'] == l].reset_index().index.to_numpy()
        x_cluster = x_subset[idx]
        cluster_mean = np.mean(x_cluster, axis=0)
        out = float(np.mean(cosine_similarity([cluster_mean], x_cluster).flatten()))
        vals += [{l: out}]

    st.write(vals)
