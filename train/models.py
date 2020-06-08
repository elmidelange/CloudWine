import pickle

from sklearn.feature_extraction.text import TfidfVectorizer

from utils import logger

# TF-IDF model training class
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
