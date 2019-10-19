from DataCleaner import DataCleaner
from Tokenizer import Tokenizer as tk
import gensim
import warnings
warnings.filterwarnings(action='ignore')

tok = tk('./Dataset/lyrics15LIN.csv', ['english', 'spanish'])

model = gensim.models.Word2Vec(tok.tokenize(), min_count=1, size=300,
                                window=5, sg=1)
print(model.similarity('man', 'something'))

b = (model['king'] - model['man'] + model['woman'])
print(model.similar_by_vector(b))
b = (model['man'] - model['money'] - model['home'])
print(model.similar_by_vector(b))