import numpy as np
from numpy import array
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf

import string, os, pdb, glob, pickle
from PIL import Image
from time import time
from tqdm import tqdm
from tensorflow.keras import Input, layers, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence, image
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Flatten, Reshape, Dropout, Bidirectional, add
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, load_img, img_to_array

token_path = "./content/Flickr8k_text/Flickr8k.token.txt"
train_images_path = './content/Flickr8k_text/Flickr_8k.trainImages.txt'
test_images_path = './content/Flickr8k_text/Flickr_8k.testImages.txt'
images_path = './content/Flickr8k_Dataset/Flicker8k_Dataset'

INCEPTION_VEC = 2048

doc = open(token_path,'r').read()

######## descriptions (dict) ########
descriptions = dict()
for line in doc.split('\n'):
    tokens = line.split()
    if len(line) > 2:
        image_id = tokens[0].split('.')[0]
        image_desc = ' '.join(tokens[1:])
        if image_id not in descriptions:
            descriptions[image_id] = list()
        descriptions[image_id].append(image_desc)

"""       
pic = '1000268201_693b08cb0e.jpg'
x=plt.imread(images_path+'/'+pic)
plt.imshow(x)
plt.show()
descriptions['1000268201_693b08cb0e']
"""

######## vocabulary (set) ########            
vocabulary = set()
for key in descriptions.keys():
    [vocabulary.update(d.split()) for d in descriptions[key]]

######## new_descriptions (string) ########
lines = list()
for key, desc_list in descriptions.items():
    for desc in desc_list:
        lines.append(key + ' ' + desc)
new_descriptions = '\n'.join(lines)

######## train (set) ########
doc = open(train_images_path,'r').read()
dataset = list()
for line in doc.split('\n'):
    if len(line) > 1:
        identifier = line.split('.')[0]
        dataset.append(identifier)
train = set(dataset)

######## train_img, test_img (list) ########    
img = glob.glob(images_path + '/*.jpg')
train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
train_img = []
for i in img:
    if i.split('/')[-1] in train_images:
        train_img.append(i)

test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
test_img = []
for i in img: 
    if i.split('/')[-1] in test_images: 
        test_img.append(i)
    
######## train_descriptions (dict) ########    
train_descriptions = dict()
for line in new_descriptions.split('\n'):
    tokens = line.split()
    image_id, image_desc = tokens[0], tokens[1:]
    if image_id in train:
        if image_id not in train_descriptions:
            train_descriptions[image_id] = list()
        desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
        train_descriptions[image_id].append(desc)

######## vocab (list) ########    
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)   

word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
        
vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

######## ixtoword, wordtoix (dict), vocab_size ########
ixtoword = {}
wordtoix = {}
ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

vocab_size = len(ixtoword) + 1

######## max_length ########    
all_desc = list()
for key in train_descriptions.keys():
    [all_desc.append(d) for d in train_descriptions[key]]
lines = all_desc
max_length = max(len(d.split()) for d in lines)

######## embedding_index (dict), embedding_matrix (numpy) ########
embeddings_index = {} 
glove_path = './content'
f = open(os.path.join(glove_path, 'glove.6B.200d.txt'), encoding="utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    
embedding_dim = 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in wordtoix.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

#######################################
################ model ################
#######################################

def preprocess(image_path):
    img = load_img(image_path, target_size=(299, 299))
    # img = image.load_img(image_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) 
    fea_vec = model_new.predict(image, verbose=0) 
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1])
    return fea_vec

def data_generator(train_descriptions, train_features, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in train_descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = train_features.get(images_path + '/' + key + '.jpg')
            if photo is None:
                n-=1
                continue
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)

            if n==num_photos_per_batch:
                yield ((array(X1), array(X2)), array(y))
                X1, X2, y = list(), list(), list()
                n=0

def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break

    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def beam_search_predictions(image, beam_index = 3):
    start = [wordtoix["startseq"]]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_length:
        temp = []
        for se in start_word:
            par_caps = pad_sequences([se[0]], maxlen=max_length, padding='post')
            preds = model.predict([image, par_caps], verbose=0)
            word_preds = np.argsort(preds[0])[-beam_index:]
            # Getting the top <beam_index>(n) predictions and creating a 
            # new list so as to put them via the model again
            for word in word_preds:
                next_cap, prob = se[0][:], se[1]
                next_cap.append(word)
                prob += preds[0][word]
                temp.append([next_cap, prob])
                    
        start_word = temp
        # Sorting according to the probabilities
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Getting the top words
        start_word = start_word[-beam_index:]
    
    start_word = start_word[-1][0]
    intermediate_caption = [ixtoword[i] for i in start_word]
    final_caption = []
    
    for i in intermediate_caption:
        if i != 'endseq':
            final_caption.append(i)
        else:
            break

    final_caption = ' '.join(final_caption[1:])
    return final_caption
        
model = InceptionV3(weights='imagenet')
# model.summary()
model_new = Model(model.input, model.layers[-2].output)        

pickle_train_path = 'encoded_train.pkl'
pickle_test_path = 'encoding_test.pkl'

if os.path.exists(pickle_train_path) and os.path.exists(pickle_test_path):
    with open(pickle_train_path, 'rb') as f:
        encoding_train = pickle.load(f)
        train_features = encoding_train
    with open(pickle_test_path, 'rb') as f:
        encoding_test = pickle.load(f)    
else:
    train_img = train_img[0:2000]
    test_img = test_img[0:500]

    encoding_train = {}
    for img in tqdm(train_img):
        encoding_train[img] = encode(img)
    train_features = encoding_train
    with open(pickle_train_path, 'wb') as f:
        pickle.dump(encoding_train, f)

    encoding_test = {}
    for img in tqdm(test_img):
        encoding_test[img] = encode(img)        
    with open(pickle_test_path, 'wb') as f:
        pickle.dump(encoding_test, f)  
    
inputs1 = Input(shape=(INCEPTION_VEC,), name="image_input")
fe1 = Dropout(0.5, name="image_dropout")(inputs1)
fe2 = Dense(256, activation='relu', name="image_dense")(fe1)

inputs2 = Input(shape=(max_length,), name="text_input")
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True, name="text_embedding")(inputs2)
se2 = Dropout(0.5, name="text_dropout")(se1)
se3 = LSTM(256, name="text_LSTM")(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.get_layer("text_embedding").set_weights([embedding_matrix])
model.get_layer("text_embedding").trainable = False  
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()

save_path = "weights_ic.weights.h5"
if os.path.exists(save_path):
    model.load_weights(save_path)
else:
    epochs = 15
    batch_size = 3
    steps = len(train_descriptions)//batch_size
    
    dataset_generator = partial(
        data_generator, 
        train_descriptions=train_descriptions,
        train_features=train_features,
        wordtoix=wordtoix,
        max_length=max_length,
        num_photos_per_batch=batch_size
    )
    
    output_signature = (
        (
            tf.TensorSpec(shape=(None, INCEPTION_VEC), dtype=tf.float32),       # X1: image features
            tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)    # X2: input sequence
        ),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)      # y: output word
    )
    
    train_dataset = tf.data.Dataset.from_generator(
        dataset_generator,
        output_signature=output_signature
    )
    
    # generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps, verbose=1)
    model.save_weights(save_path)

fig_path = "output_image.png"
if os.path.exists(fig_path):
    os.remove(fig_path)

pic = list(encoding_test.keys())[20]
image = encoding_test[pic].reshape((1, INCEPTION_VEC))
x=plt.imread(pic)
plt.imshow(x)
plt.axis('off')
plt.savefig(fig_path)

# print("Greedy:",greedySearch(image))
print("Beam Search, K = 3:",beam_search_predictions(image, beam_index = 3))
print("Beam Search, K = 5:",beam_search_predictions(image, beam_index = 5))
# print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))