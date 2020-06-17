import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from cloudwine.utils import logger


def preprocess(text):
    # Convert to lowercase
    out_text = text.lower()
    # Remove puctuation
    out_text = out_text.replace('[^\w\s]','')
    # Remove stopwords
    stops = stopwords.words('english')
    word_list = out_text.split()
    meaningful_words = [w for w in word_list if not w in stops]
    out_text = ' '.join(meaningful_words)
    # Lemmatisation
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)


# Loads TF-IDF model and inferences input string
def inference_tfidf(data_module, text):
    logger.info('TF-IDF Inference')
    # Load trained model pickles
    tf = data_module.model
    x = data_module.vectors
    df = data_module.data_filtered
    # Get filtered embeddings
    x = x[df.index.tolist()]

    # Create new tfidfVectorizer with trained vocabulary
    tf_new = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True,
                          max_features = 500000, vocabulary = tf.vocabulary_)

    # Convert text to vector representation
    x_new = tf_new.fit_transform([text])

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    df_subset = df.reset_index().loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset


# Loads TF-IDF model and inferences input string
def inference_docvec(data_module, text):
    logger.info('Doc2Vec Inference')
    # Load trained model pickles
    model = data_module.model
    x = data_module.vectors
    df = data_module.data_filtered
    # Get filtered embeddings
    x = x[df.index.tolist()]

    words=word_tokenize(text.lower())
    x_new = model.infer_vector(words)

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity([x_new], x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    df_subset = df.reset_index().loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset


# Loads BERT embeddings and inferences input string
def inference_bert(data_module, text):
    logger.info('BERT Inference')
    # Load trained model pickles
    model = data_module.model
    x = data_module.vectors
    df = data_module.data_filtered
    # Get filtered embeddings
    x = x[df.index.tolist()]
    # Inference new text
    x_new = model.encode([text])

    # Calculate cosine similarities to trained text
    cosine_similarities = cosine_similarity(x_new, x).flatten()

    # Get the index to the top 3 similar texts
    related_docs_indices = cosine_similarities.argsort()[:-6:-1]

    df_subset = df.reset_index().loc[related_docs_indices].reset_index(drop=True)
    df_subset['similarity'] = cosine_similarities[related_docs_indices]
    return df_subset
