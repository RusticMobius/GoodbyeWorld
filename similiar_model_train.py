from gensim.models import word2vec
from gensim.test.utils import common_texts, get_tmpfile
import logging


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


sentences = word2vec.LineSentence('sim_data.csv')
path = get_tmpfile("word2vec.model")
model = word2vec.Word2Vec(sentences, hs=1, min_count=3, window=5, workers=5)
model.save("word2vec.model")