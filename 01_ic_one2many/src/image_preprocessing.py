import os
import pickle
import glob 
import pdb
import traceback
import numpy as np

from tqdm import tqdm

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import Model

### train (set) ###
def remove_jpg(doc):
    dataset = list()
    for line in doc.split('\n'):
        if len(line) > 1:
            identifier = line.split('.')[0]
            dataset.append(identifier)
    train = set(dataset)
    return train

### train_img, test_img (list) ###
def check_imagefile(images_path, train_images_path, test_images_path):
    img = glob.glob(images_path + '\*.jpg')
    train_images = set(open(train_images_path, 'r').read().strip().split('\n'))
    train_img = []
    for i in img:
        if i.split("\\")[-1] in train_images:
            train_img.append(i)

    test_images = set(open(test_images_path, 'r').read().strip().split('\n'))
    test_img = []
    for i in img: 
        if i.split("\\")[-1] in test_images: 
            test_img.append(i)
    return train_img, test_img


def make_image_embedding(train_img, test_img, pickle_train_path='encoded_train.pkl', pickle_test_path='encoding_test.pkl'):
    model = InceptionV3(weights='imagenet')
    # model.summary()
    model_new = Model(model.input, model.layers[-2].output)   

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

    if os.path.exists(pickle_train_path) and os.path.exists(pickle_test_path):
        with open(pickle_train_path, 'rb') as f:
            encoding_train = pickle.load(f)
            train_features = encoding_train
        with open(pickle_test_path, 'rb') as f:
            encoding_test = pickle.load(f) 
            test_features = encoding_test 
    else:
        train_img = train_img[0:2000]
        test_img = test_img[0:500]

        encoding_train = {}
        for img in tqdm(train_img):
            try:
                encoding_train[img] = encode(img)
            except Exception as e:
                print(f"[ERROR] Failed to encode image: {img}")
                traceback.print_exc()
            # encoding_train[img] = encode(img)
        train_features = encoding_train
        with open(pickle_train_path, 'wb') as f:
            pickle.dump(encoding_train, f)

        encoding_test = {}
        for img in tqdm(test_img):
            try:
                encoding_test[img] = encode(img) 
            except Exception as e:
                print(f"[ERROR] Failed to encode image: {img}")
                traceback.print_exc()
            # encoding_test[img] = encode(img)  
            test_features = encoding_test      
        with open(pickle_test_path, 'wb') as f:
            pickle.dump(encoding_test, f)  
    return train_features, test_features




