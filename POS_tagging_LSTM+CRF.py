import nltk, re, pprint
import numpy as np
#import numpy as np
from tqdm import tqdm
import pandas as pd
import requests
#import matplotlib.pyplot as plt
#import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn_crfsuite import CRF
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
#from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader
corpusdir ='_DATA' # Directory of corpus.
#newcorpus = PlaintextCorpusReader(corpusdir, '.*')
newcorpus = TaggedCorpusReader(corpusdir, '.*')
print(newcorpus.fileids())
dictlist=[]

#Converting from word/tag pait to a list containing two tuple i.e,[(word1,tag),(word2,tag)]
for i in newcorpus.fileids():
    tagged_sent=newcorpus.raw(i)
    tagged=tagged_sent.split()
    for t in tagged:
        temp1=nltk.tag.str2tuple(t)
        dictlist.append(temp1) 
tagged_sentence = newcorpus.tagged_sents()
#print(dictlist)


tagged_words=[tup for sent in tagged_sentence for tup in sent]
words=[word for word,tag in tagged_words]
words.append("ENDPAD")
print("Total Number of Tagged words", len(words))
n_words = len(tagged_words)

tags1=set([tag for word,tag in tagged_words])
print("Number of Tags in the Corpus ",len(tags1))
n_tags = len(tags1)
print("Number of Tagged Sentences ",len(tagged_sentence))
sentences=[tup for sent in tagged_sentence for tup in sent]
print("Total Number of Tagged words in sentences", len(sentences))

from collections import Counter, defaultdict
word_counts = Counter()
#for sentence in tagged_sentence:
 #   w, t = zip(*sentence)
  #  word_counts.update(w)

    
    
max_len = 75
#word_to_idx = defaultdict(lambda:1, {words:idx for idx,words in tqdm(enumerate(vocab))})
word2idx = defaultdict(lambda:1,{w: i + 1 for i, w in tqdm(enumerate(words))})
#tag_to_idx = {tag:idx for idx,tag in tqdm(enumerate(all_tags))}
tag2idx = {t:i for i, t in tqdm(enumerate(tags1))}

#max_len = 75
#word2idx = {w: i + 1 for i, w in enumerate(words)}
#tag2idx = {t: i for i, t in enumerate(tags1)}


import keras
import keras.layers as L
import sys
from keras.utils.np_utils import to_categorical
from keras.callbacks import LambdaCallback
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical


X = [[word2idx[word] for word,tag in s] for s in tagged_sentence]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

y = [[tag2idx[tag] for word,tag in s] for s in  tagged_sentence]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)

y = [to_categorical(i, num_classes=n_tags) for i in y]


from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional,Conv2D
from keras_contrib.layers import CRF
input = Input(shape=(max_len,))

model = Embedding(input_dim=n_words + 1, output_dim=20,input_length=max_len)(input) # 20-dim embedding
#model = Conv2D(filters=32, kernel_size=4, activation='relu')(model)
#model=Conv2D(64, kernel_size=3, activation="relu", input_shape=max_len)
#model=Conv2D(32, kernel_size=3, activation="relu")
#model=add(Conv2D(64, kernel_size=3, activation="relu", input_shape=max_len)(model)
#model=Conv2D(32, kernel_size=3, activation="relu")
model = Bidirectional(LSTM(units=50, return_sequences=True,recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
#TimeDistributed(L.Dense(len(all_tags),activation='softmax'))
crf = CRF(n_tags)  
out = crf(model) 
model = Model(input, out)
model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()
history = model.fit(X_tr, np.array(y_tr), batch_size=32, epochs=50,
                    validation_split=0.1, verbose=1)
