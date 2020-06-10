import pickle
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.tokenize import word_tokenize

from sentence_transformers import SentenceTransformer

from utils import logger

# TF-IDF model training class
class TfidfTrainer:
    def __init__(self):
        self.tf_transformer = None
        self.text_vectors = None

    # Train TF-IDF model
    def train(self, corpus):
        logger.info("Training TF-IDF model")
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), stop_words = "english", lowercase = True, max_features = 768)
        self.tf_transformer = tf.fit(corpus)
        self.text_vectors = tf.fit_transform(corpus).toarray()

    # Returns the trained sentence vectors
    def get_vectors(self):
        return self.text_vectors

    # Save the model in pickle format
    def save(self, path):
        logger.info("Saving model to %s", path)
        # Save TfidfVectoriser vocab
        with open(path + '/tfidf_model.pkl', "wb") as pickleFile:
            pickle.dump(self.tf_transformer, pickleFile)
        # Save corpus text vectors
        logger.info("Saving vectors to %s", path)
        with open(path +  '/tfidf_vectors.pkl', "wb") as pickleFile:
            pickle.dump(self.text_vectors, pickleFile)


# Doc2Vec model training class
class DocVecTrainer:
    def __init__(self):
        self.model = None
        self.text_vectors = None
        self.vec_size = 768
        self.max_epochs = 10

    # Train Doc2Vec model
    def train(self, corpus):
        logger.info("Training Doc2Vec model")
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(corpus)]
        alpha = 0.025
        # Create model
        model = Doc2Vec(size=self.vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=1,
                        dm =1)
        model.build_vocab(tagged_data)
        # Train model
        for epoch in range(self.max_epochs):
            logger.info('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        self.model = model
        self.text_vectors = self.get_vectors()

    # Returns the trained sentence vectors
    def get_vectors(self):
        docvecs = self.model.docvecs['0']
        for i in range(1,self.model.corpus_count):
            docvecs = np.vstack((docvecs, self.model.docvecs[str(i)]))
        return docvecs

    # Save the model in pickle format
    def save(self, path):
        logger.info("Saving model to %s", path)
        # Save model
        with open(path + '/doc2vec_model.pkl', "wb") as pickleFile:
            pickle.dump(self.model, pickleFile)
        # Save corpus text vectors
        logger.info("Saving vectors to %s", path)
        with open(path +  '/doc2vec_vectors.pkl', "wb") as pickleFile:
            pickle.dump(self.text_vectors, pickleFile)


# BERT model training class
class BertTrainer:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.text_vectors = None

    # Embed corpus with BERT embeddings
    def train(self, corpus):
        logger.info("Training BERT model")
        sentence_embeddings = self.model.encode(corpus)
        self.text_vectors = np.array(sentence_embeddings)

    # Returns the trained sentence vectors
    def get_vectors(self):
        return self.text_vectors

    # Save the model in pickle format
    def save(self, path):
        logger.info("Saving model to %s", path)
        # Save model
        with open(path + '/bert_model.pkl', "wb") as pickleFile:
            pickle.dump(self.model, pickleFile)
        # Save corpus text vectors
        logger.info("Saving vectors to %s", path)
        with open(path +  '/bert_vectors.pkl', "wb") as pickleFile:
            pickle.dump(self.text_vectors, pickleFile)
