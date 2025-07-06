import os, json
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, 
    Dropout, 
    TimeDistributed, 
    Dense, 
    Activation, 
    Embedding, 
    Input,
    )

#
# load data
#

with open("input.txt", "r") as file:
    text = file.read()

#
# preprocessing
#

char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

with open('char_to_idx.json', 'w') as f:
    json.dump(char_to_idx, f)

#
# train
#

def build_model(batch_size, seq_len, vocab_size):
    model = Sequential()
    model.add(Input(batch_shape=(batch_size, seq_len)))
    model.add(Embedding(input_dim=vocab_size, output_dim=512, input_length=seq_len))
    for i in range(4):
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(vocab_size))) 
    model.add(Activation('softmax'))
    return model

BATCH_SIZE = 207
SEQ_LENGTH = 64

model = build_model(BATCH_SIZE, SEQ_LENGTH, vocab_size)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32) 
steps_per_epoch = (len(text) / BATCH_SIZE - 1) / SEQ_LENGTH  

def read_batches(T, vocab_size):
    length = T.shape[0]
    batch_chars = int(length / BATCH_SIZE)
    for start in range(0, batch_chars - SEQ_LENGTH, SEQ_LENGTH):
        X = np.zeros((BATCH_SIZE, SEQ_LENGTH))
        Y = np.zeros((BATCH_SIZE, SEQ_LENGTH, vocab_size))
        for batch_idx in range(0, BATCH_SIZE):
            for i in range(0, SEQ_LENGTH):
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i]
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y

save_path = 'weights_new.weights.h5'
if os.path.exists(save_path):
    model.load_weights(save_path)
else:    
    epochs = 20
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        losses, accs = [], []
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            loss, acc = model.train_on_batch(X, Y)
            losses.append(loss)
            accs.append(acc)
            print(f"loss: {loss}, accuracy: {acc}")     
    model.save_weights(save_path)

#
# inference
#

# 예측용 모델 (batch_size=None으로 Stateless 모델)
inference_model = build_model(batch_size=1, seq_len=1, vocab_size=vocab_size)
inference_model.set_weights(model.get_weights())  # 기존 학습된 가중치 복사
    
num_chars = 256
header = ''

with open('char_to_idx.json') as f:
    char_to_idx = json.load(f)

idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

sampled = [char_to_idx[c] for c in header]

for i in range(num_chars):
    batch = np.zeros((1, 1))
    if sampled:
        batch[0, 0] = sampled[-1]
    else:
        batch[0, 0] = np.random.randint(vocab_size)
        
    result = inference_model.predict_on_batch(batch).ravel()
    sample = np.random.choice(range(vocab_size), p=result)
    sampled.append(sample)

generated_text = ''.join([idx_to_char[i] for i in sampled])
print(generated_text)   