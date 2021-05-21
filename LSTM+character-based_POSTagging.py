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

max_len_char = 15

from collections import Counter, defaultdict
max_len = 75
w2idx = defaultdict(lambda:1,{w: i + 1 for i, w in tqdm(enumerate(words))})
#w2idx["UNK"] = 1
w2idx["PAD"] = 0
t2idx  = {t:i for i, t in tqdm(enumerate(tags1))}
t2idx ["PAD"] = 0
idx2tag = {i: w for w, i in t2idx .items()}
from keras.preprocessing.sequence import pad_sequences
X_word = [[w2idx[w[0]] for w in s] for s in tagged_sentence]
X_word = pad_sequences(maxlen=max_len, sequences=X_word, value=w2idx["PAD"], padding='post', truncating='post')
chars = set([w_i for w in words for w_i in w])
n_chars = len(chars)
print(n_chars)
c2idx = {c: i + 2 for i, c in enumerate(chars)}
c2idx["UNK"] = 1
c2idx["PAD"] = 0
X_char = []
for sentence in tagged_sentence:
    sent_seq = []
    for i in range(max_len):
        word_seq = []
        for j in range(max_len_char):
            try:
                word_seq.append(c2idx.get(sentence[i][0][j]))
            except:
                word_seq.append(c2idx.get("PAD"))
        sent_seq.append(word_seq)
    X_char.append(np.array(sent_seq))
y = [[t2idx [tag] for word,tag in s] for s in  tagged_sentence]
#y = pad_sequences(maxlen=max_len, sequences=y, value=t2idx ["PAD"], padding='post', truncating='post')
y = pad_sequences(maxlen=max_len, sequences=y,  padding='post', truncating='post')    
 from sklearn.model_selection import train_test_split
#X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)
X_word_tr, X_word_te, y_tr, y_te = train_test_split(X_word, y, test_size=0.2, random_state=2018)
X_char_tr, X_char_te, _, _ = train_test_split(X_char, y, test_size=0.2, random_state=2018)
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Conv1D
from keras.layers import Bidirectional, concatenate, SpatialDropout1D, GlobalMaxPooling1D
# input and embedding for words
word_in = Input(shape=(max_len,))
emb_word = Embedding(input_dim=n_words + 2, output_dim=20,
                     input_length=max_len, mask_zero=True)(word_in)

# input and embeddings for characters
char_in = Input(shape=(max_len, max_len_char,))
emb_char = TimeDistributed(Embedding(input_dim=n_chars + 2, output_dim=10,
                           input_length=max_len_char, mask_zero=True))(char_in)
# character LSTM to get word encodings by characters
char_enc = TimeDistributed(LSTM(units=20, return_sequences=False,
                                recurrent_dropout=0.5))(emb_char)

# main LSTM
x = concatenate([emb_word, char_enc])
x = SpatialDropout1D(0.3)(x)
main_lstm = Bidirectional(LSTM(units=50, return_sequences=True,
                               recurrent_dropout=0.6))(x)
out = TimeDistributed(Dense(n_tags + 1, activation="sigmoid"))(main_lstm)
model = Model([word_in, char_in], out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])
model.summary()

