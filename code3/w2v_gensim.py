import gensim
from tqdm import tqdm

config = {
    'window_size': 2,
    'epochs': 10,
    'vector_size': 300
}


model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count,
            epochs=model.epochs)
