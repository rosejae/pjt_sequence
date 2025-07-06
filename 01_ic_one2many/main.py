import pdb
import os
import matplotlib.pyplot as plt

from src import *

base_path = r"C:\\Users\\jjsd4\\workspace\\18_30_sequencial\\ch2\\content"
### cap-img ###
token_path = os.path.join(base_path, r"Flickr8k_text\\Flickr8k.token.txt")
### img ###
images_path = os.path.join(base_path, r"Flickr8k_Dataset\\Flicker8k_Dataset")
train_images_path = os.path.join(base_path, r"Flickr8k_text\\Flickr_8k.trainImages.txt")
test_images_path = os.path.join(base_path, r"Flickr8k_text\\Flickr_8k.testImages.txt")
### image embedding dimensions ###
INCEPTION_VEC = 2048

# pickle_train_path = 'encoded_train.pkl'
# pickle_test_path = 'encoding_test.pkl'

if __name__ == '__main__':

    #####################
    ### preprocessing ###
    #####################
    
    ### img-cap textfile load ###
    cap_doc = open(token_path,'r').read()
    descriptions, vocabulary, new_descriptions = textfile_preprocessing(cap_doc)

    ### img textfile load ###
    img_doc = open(train_images_path,'r').read()  
    train = remove_jpg(img_doc)
    train_img, test_img = check_imagefile(images_path, train_images_path, test_images_path)

    ### caption preprocessing ###
    train_descriptions = add_startseq_endseq(new_descriptions, train)

    ### make vocab and text embedding matrix ###
    vocab, wordtoix, ixtoword, vocab_size, max_length = make_vocab(train_descriptions)
    embedding_matrix = make_text_embedding(vocab_size, wordtoix, base_path)

    ### make image embedding vector ###
    train_features, test_features = make_image_embedding(train_img, test_img)

    #####################
    ####### train #######
    #####################

    model = train_model(
        train_descriptions, 
        train_features, 
        wordtoix, 
        max_length, 
        vocab_size, 
        embedding_matrix,
        images_path,
    )

    #########################
    ####### inference #######
    #########################

    fig_path = "output_image.png"
    if os.path.exists(fig_path):
        os.remove(fig_path)

    pic = list(test_features.keys())[20]
    image = test_features[pic].reshape((1, INCEPTION_VEC))
    x=plt.imread(pic)
    plt.imshow(x)
    plt.axis('off')
    plt.savefig(fig_path)

    # print("Greedy:",greedySearch(image))
    print("Beam Search, K = 3:", beam_search_predictions(image, model, wordtoix, ixtoword, max_length, beam_index = 3))
    print("Beam Search, K = 5:", beam_search_predictions(image, model, wordtoix, ixtoword, max_length, beam_index = 5))
    # print("Beam Search, K = 7:",beam_search_predictions(image, beam_index = 7))