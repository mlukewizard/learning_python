# from gensim.models import Word2Vec, doc2vec
# from gensim.models import KeyedVectors
# from gensim.test.utils import datapath
from sklearn.cluster import KMeans
import numpy as np
import spacy

nlp_model = spacy.load('en')
# nlp_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

sentences = [...]
tokenised_sentences = [nlp_model(sentence) for sentence in sentences]


words = [...]

item_vectors = np.array([item.vector for item in tokenised_sentences])
cluster_model = KMeans(n_clusters=12)
cluster_model = cluster_model.fit(item_vectors)
classes = cluster_model.predict(item_vectors)
dict = {}
for i in range(np.max(classes) + 1):
    dict[i] = [item for item, _class in zip(sentences, classes) if _class == i]
