
# coding: utf-8

# In[40]:

import gensim
from gensim.models.word2vec import Word2Vec


def train():
    sentences = gensim.models.word2vec.PathLineSentences("/floyd/input/corpus/RESULT")
    print("Corpus file: ", sentences.input_files)

    word2vec_model = Word2Vec(sentences=sentences, 
            size=100, window=5, min_count=1, workers=4, sg=0)
    print("Similar to ưng_hoàng_phúc: ", word2vec_model.wv.most_similar("ưng_hoàng_phúc"))
    word2vec_model.save("song2vec.model")


if __name__ == '__main__':
    train()







