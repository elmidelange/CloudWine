import pickle
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

def main():
    model_dir = './models/'

    x_tfidf = pickle.load(open(model_dir + "tfidf_vectors.pkl", 'rb'))
    x_tfidf = x_tfidf / x_tfidf.max()
    x_docvec = pickle.load(open(model_dir + "doc2vec_vectors.pkl", 'rb'))
    x_docvec = x_docvec / x_docvec.max()
    x_bert = pickle.load(open(model_dir + "bert_vectors.pkl", 'rb'))
    x_bert = x_bert / x_bert.max()

    x_all = np.concatenate((x_tfidf, x_docvec, x_bert))

    # Perform t-SNE
    pca = PCA(n_components=2)
    pca_result_all = pca.fit_transform(x_all)

    print(pca_result_all.shape)

    length_tfidf = x_tfidf.shape[0]
    length_docvec = x_docvec.shape[0]
    length_bert = x_bert.shape[0]
    # Load trained embeddings
    pca_result_tfidf = pca_result_all[:length_tfidf]
    pca_result_docvec = pca_result_all[length_tfidf:length_tfidf + length_docvec]
    pca_result_bert = pca_result_all[length_tfidf+length_docvec:length_tfidf + length_docvec + length_bert]

    data = np.append(np.append(pca_result_tfidf, pca_result_docvec, axis=1), pca_result_bert, axis=1)

    df = pd.DataFrame(data, columns = ['tfidf_1', 'tfidf_2', 'doc2vec_1', 'doc2vec_2', 'bert_1', 'bert_2'])

    df.to_pickle(model_dir + 'pca_2d.pkl')


if __name__ == "__main__":
    main()
