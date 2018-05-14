import gensim
import nltk
from nltk.corpus import stopwords
import logging
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# with open('data/nietzsche.txt') as f:
#     document = f.read()
# print(document)

# stop_words = set(stopwords.words('english'))
# print(stop_words)


#for LineSentence fucntion, each line will be recgonized as a sentence.#
sentences = word2vec.LineSentence('data/nietzsche.txt')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=5,size=100)

print(model.most_similar(['man']))
print(model.most_similar(positive=['woman'], negative=['man'], topn=2)  )
# req_count = 5
# for key in model.wv.similar_by_word('opposite', topn =100):
#     if len(key[0])==3:
#         req_count -= 1
#         print (key[0], key[1])
#         if req_count == 0:
#             break;