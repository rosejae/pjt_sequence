import string, os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

#
# load data
#

def read_text(filename):
        file = open(filename, mode='rt', encoding='utf-8')
        text = file.read()
        file.close()
        return text

data = read_text(r".\dataset\deu.txt")

#
# preprocessing
#

def to_lines(text):
    sents = text.strip().split('\n')
    sents = [i.split('\t') for i in sents]
    return sents

deu_eng = to_lines(data)
deu_eng = np.array(deu_eng)

deu_eng[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,0]]
deu_eng[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in deu_eng[:,1]]

for i in range(len(deu_eng)):
    deu_eng[i,0] = deu_eng[i,0].lower()
    deu_eng[i,1] = deu_eng[i,1].lower()

eng_l = []
deu_l = []

for i in deu_eng[:,0]:
    eng_l.append(len(i.split()))

for i in deu_eng[:,1]:
    deu_l.append(len(i.split()))

#
# tokenize
#

def tokenization(lines, vocab_limit=None): 
    # due to the gpu resource, num_words and oov_token is included
    tokenizer = Tokenizer(num_words=vocab_limit, oov_token='<OOV>')
    tokenizer.fit_on_texts(lines)
    return tokenizer

eng_tokenizer = tokenization(deu_eng[:, 0], vocab_limit=10000)
# eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_vocab_size = 10000
eng_length = 8

deu_tokenizer = tokenization(deu_eng[:, 1], vocab_limit=10000)
# deu_vocab_size = len(deu_tokenizer.word_index) + 1
deu_vocab_size = 10000
deu_length = 8

def encode_sequences(tokenizer, length, lines):
    seq = tokenizer.texts_to_sequences(lines)
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

train, test = train_test_split(deu_eng, test_size=0.2, random_state = 12)

trainX = encode_sequences(deu_tokenizer, deu_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])

testX = encode_sequences(deu_tokenizer, deu_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])

#
# train
#

def define_model(in_vocab, out_vocab, in_timesteps, out_timesteps, units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dense(out_vocab, activation='softmax'))
    return model

model = define_model(deu_vocab_size, eng_vocab_size, deu_length, eng_length, 128)

rms = optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

save_path = 'model.keras'
if os.path.exists(save_path):
    model = load_model('model.keras')
else:
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    history = model.fit(trainX, 
                        trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                        epochs=40, 
                        batch_size=256, 
                        validation_split=0.2, 
                        callbacks=[checkpoint], 
                        verbose=1,
                        )
    
#
# inference
#
    
# def get_word(n, tokenizer):
#     for word, index in tokenizer.word_index.items():
#         if index == n:
#             return word
#     return None
    
def get_word(index, tokenizer):
    return tokenizer.index_word.get(index, '')
    
# pdb.set_trace()
testX1 = testX[0:5]
    
preds = model.predict(testX1)
preds = np.argmax(preds, axis=-1)    

decoded_preds = []
for seq in preds:
    words = [get_word(i, eng_tokenizer) for i in seq]
    sentence = ' '.join([w for w in words if w])
    decoded_preds.append(sentence)
    
for i in range(len(decoded_preds)):
    print(f"원본(독일어): {test[i, 1]}")
    print(f"정답(영어):   {test[i, 0]}")
    print(f"예측(모델):   {decoded_preds[i]}")
    print("-" * 50)

# preds_text = []

# for i in preds:
#     temp = []
#     for j in range(len(i)):
#         t = get_word(i[j], eng_tokenizer)
#         if j > 0:
#             if (t == get_word(i[j-1], eng_tokenizer)) or (t == None):
#                 temp.append('')
#             else:
#                 temp.append(t)
#         else:
#             if(t == None):
#                 temp.append('')
#             else:
#                 temp.append(t) 

#     preds_text.append(' '.join(temp))