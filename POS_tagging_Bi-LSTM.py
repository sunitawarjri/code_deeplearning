




import nltk
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus.reader.tagged import TaggedCorpusReader 
path = '_DATA'
corpus1 = TaggedCorpusReader(path, '.*')
print(corpus1.fileids())
data = corpus1.tagged_sents()
data = []
n_tags = 53
for i in corpus1.fileids():
    #send_tag = corpus1.raw(i)
    send_tag = corpus1.tagged_sents(i)
    #tagged = send_tag.split()
    ######
    #for j in tagged:
    #    a = nltk.tag.str2tuple(j)
     #   data.append(a)
        
train_data = corpus1.tagged_sents()

path1 = 'TRy_train_test'
corpus2 = TaggedCorpusReader(path1, '.*')
print(corpus2.fileids())
test_data = corpus2.tagged_sents()
#train_data, test_data = train_test_split(data,test_size=0.20)
print(len(train_data))
print(len(test_data))
from collections import Counter, defaultdict
word_counts = Counter()
# we will use the top 11000 words for out dictionary only.
for sentence in train_data:
    words, tags = zip(*sentence)
    word_counts.update(words)
# take out the top words
top_words = list(zip(*word_counts.most_common(11000)))[0]
# add two other tags: <EOS>:End of Sentence, <UNK>:Unknown 
vocab = ['XX'] + list(top_words) 
#print(vocab)

# create word to index mapping
# for every unknown word the dict will give index 1 which is <UNK>
word_to_idx = defaultdict(lambda:1, {words:idx for idx,words in tqdm(enumerate(vocab))})
# create reverse mapping
idx_to_word = {idx:words for words,idx in word_to_idx.items()}

List_t = ['XX','PPN','IM','CMN', 'MTN', 'ABN', 'RFP', 'EM', 'RLP',
   'INP', 'DMP', 'IDP','POP', 'CAV', 'TRV', 'ITV', 'DTV', 'ADJ', 'CMA',
  'SPA', 'AD', 'ADT', 'ADM', 'ADP', 'ADF', 'ADD', 'IN',
 '1PSG', '1PPG', '2PG', '2PF', '2PM', '3PSF', '3PSM', '3PPG', 
 '3PSG', 'VPT', 'VPP', 'VST','VSP', 'VFT', 'MOD', 'NEG', 'CLF', 
'.', 'COC', 'SUC', 'CRC', 'CN', 'ON', 'QNT', 'CO', 'PAV', 'COM', 'FR', 'SYM','CLN']

# create word to index mapping
tag_to_idx = {tag:idx for idx,tag in tqdm(enumerate(List_t))}
# create reverse mapping
idx_to_tag = {idx:tags for tags,idx in tag_to_idx.items()}

# converts the tokens to its numerical representation
# output: (m, max sequence length)
def convert_to_num(sentences, token_to_idx, pad=0, dtype='int32'):
    # find the max sentence length
    max_sent_len = max(map(len, sentences))
    
    # create the matrix
    mat = np.empty([len(sentences), max_sent_len], dtype)
    # fill with padding
    mat.fill(pad)
    
    # convert to numerical mappings
    for i, sentence in enumerate(sentences):
        num_row = [token_to_idx[token] for token in sentence]
        mat[i, :len(num_row)] = num_row
   
    return mat


words_batch, tags_batch = zip(*[zip(*sentence) for sentence in train_data[0:]])

print(convert_to_num(words_batch, word_to_idx))
print(convert_to_num(tags_batch,tag_to_idx))

import keras
import keras.layers as L
import sys
from keras.utils.np_utils import to_categorical
from keras.callbacks import LambdaCallback
import tensorflow as tf

# for generating the batches
def generate_model_batches(sentences, batch_size=16, pad=0):
    # no. of training examples
    m = np.arange(len(sentences))
    
    while True:
        # get a shuffled index list
        idx = np.random.permutation(m)
        
        # start yeilding batches
        for start in range(0, len(idx)-1, batch_size):
            batch_idx = idx[start:start+batch_size]
            batch_words, batch_tags = [], []
            
            # take out the words and tags from 'batch_size' no. of training examples
            for index in batch_idx:
                words, tags = zip(*sentences[index])
                batch_words.append(words)
                batch_tags.append(tags)
            
            # input x
            batch_words_num = convert_to_num(batch_words, word_to_idx, pad=0)
            batch_tags_num = convert_to_num(batch_tags, tag_to_idx, pad=0)
            
            # output labels 
            batch_tags_ohe = to_categorical(batch_tags_num, len(List_t))
            yield batch_words_num, batch_tags_ohe
            
        # for computing accuracy
def compute_accuracy(model):
    test_words, test_tags = zip(*[zip(*sentence) for sentence in test_data])
    test_words_num = convert_to_num(test_words, word_to_idx)
    test_tags_num = convert_to_num(test_tags, tag_to_idx)
    
    # get prediction tags
    predictions = model.predict(test_words_num, batch_size=128, verbose=1)
    pred_tags = predictions.argmax(axis=-1)
    
    # compute accuracy
    return float(np.sum(np.logical_and((test_words_num!=0), (pred_tags == test_tags_num))))  \
                /np.sum(test_words_num!=0)
                # for computing accuracy at the end of epoch
def on_epoch_end(epoch, logs):
    sys.stdout.flush()
    print('\nValidation Accuracy: ' + str(compute_accuracy(model)*100) + ' %')
    sys.stdout.flush()
    

acc_callback = LambdaCallback(on_epoch_end=on_epoch_end)
model = keras.models.Sequential()
model.add(L.InputLayer([None],dtype='int32'))
model.add(L.Embedding(len(vocab),50))

model.add(L.Bidirectional(L.LSTM(64,return_sequences=True,activation='tanh')))
model.add(L.Dropout(0.35))
model.add(L.BatchNormalization())

stepwise_dense = L.TimeDistributed(L.Dense(len(List_t),activation='softmax'))
model.add(stepwise_dense)


model.summary()

adam = keras.optimizers.Adam(clipvalue=1.5)
model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

with tf.device('/gpu:0'):
    hist = model.fit_generator(generate_model_batches(train_data, batch_size=256),steps_per_epoch=len(train_data)/256,
                   callbacks=[acc_callback], epochs=50)
#print(test_data)
test_words, test_tags = zip(*[zip(*sentence) for sentence in test_data])
test_words_num = convert_to_num(test_words, word_to_idx)
test_tags_num = convert_to_num(test_tags, tag_to_idx)
print(test_tags_num)
print(model.predict(test_tags_num))  
#print('tags', model.predict(test_tags_num)) 
  # acc_callback = LambdaCallback(on_epoch_end=on_epoch_end) 
  
test_sent = ['U Myntri Rangbah ka jylla u Conrad K Sangma , u Deputy Chief Minister bah Prestone Tynsong lem bad ki myntri']
test_sent_token = [[word for word in sent.split()] for sent in test_sent]
test_sent_num = convert_to_num(test_sent_token, word_to_idx, pad=0)

pred = model.predict(test_sent_num)
print(pred)
for i in range(len(pred)):
    result = []
    print('Given: ',test_sent[i])
  
    for j in range(len(pred[i])):  
        index = pred[i,j,:].argmax()
        result.append(idx_to_tag[index])
    print('Prediction: ', result)
    true_res = list(zip(*nltk.pos_tag(test_sent_token[i], tagset='universal')))[1]
    #print('NLTK Tagger: ', true_res)
    print()
    
 tagged_words=[tup for sent in train_data for tup in sent]
tags1 = [tag for (word, tag) in tagged_words]
nltk.FreqDist(tags1).max()

import nltk
from nltk.probability import FreqDist
fd = nltk.FreqDist(tags1)
fd.tabulate()

