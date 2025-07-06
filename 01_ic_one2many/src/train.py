import os

import tensorflow as tf 

from functools import partial
from numpy import array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from . import o2m_model

def train_model(train_descriptions, train_features, wordtoix, max_length, vocab_size, embedding_matrix, images_path, INCEPTION_VEC=2048, save_path='weights_ic.weights.h5'):
    def data_generator(train_descriptions, train_features, wordtoix, max_length, num_photos_per_batch):
        X1, X2, y = list(), list(), list()
        n=0
        # loop for ever over images
        while 1:
            for key, desc_list in train_descriptions.items():
                n+=1
                # retrieve the photo feature
                photo = train_features.get(images_path + "\\" + key + '.jpg')
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
    
    model = o2m_model.build_model(max_length, vocab_size, embedding_matrix)
           
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
                tf.TensorSpec(shape=(None, INCEPTION_VEC), dtype=tf.float32), # X1: image features
                tf.TensorSpec(shape=(None, max_length), dtype=tf.int32)       # X2: input sequence
            ),
            tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)         # y: output word
        )
        
        train_dataset = tf.data.Dataset.from_generator(
            dataset_generator,
            output_signature=output_signature
        )
        
        # generator = data_generator(train_descriptions, train_features, wordtoix, max_length, batch_size)
        model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps, verbose=1)
        model.save_weights(save_path)
    return model
