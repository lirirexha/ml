from gensim.models import Word2Vec
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

import nltk
nltk.download('punkt')

warnings.filterwarnings(action='ignore')

sample = open("alice.txt")
s = sample.read()

f = s.replace("\n", " ")

data = []

for i in sent_tokenize(f):
    temp = []

    # tokenize the sentence into words
    for j in word_tokenize(i):
        temp.append(j.lower())

    data.append(temp)

model1 = gensim.models.Word2Vec(data, min_count=1, vector_size=100, window=5)

print("similaity between Queen and Hearts: ", model1.wv.similarity('queen', 'hearts'))
