import os
import numpy as np

### descriptions (dict) ###
def textfile_preprocessing(doc):
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        if len(line) > 2:
            image_id = tokens[0].split('.')[0]
            image_desc = ' '.join(tokens[1:])
            if image_id not in descriptions:
                descriptions[image_id] = list()
            descriptions[image_id].append(image_desc)
            
    ### vocabulary (set) ###  
    def dict_to_vocab(descriptions):
        vocabulary = set()
        for key in descriptions.keys():
            [vocabulary.update(d.split()) for d in descriptions[key]]
        return vocabulary
    
    ### new_descriptions (string) ###
    def dict_to_list(descriptions):
        lines = list()
        for key, desc_list in descriptions.items():
            for desc in desc_list:
                lines.append(key + ' ' + desc)
        new_descriptions = '\n'.join(lines)
        return new_descriptions        
    
    vocabulary = dict_to_vocab(descriptions)   
    new_descriptions = dict_to_list(descriptions)
    return descriptions, vocabulary, new_descriptions

### train_descriptions (dict) ###   
def add_startseq_endseq(new_descriptions, train): 
    train_descriptions = dict()
    for line in new_descriptions.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in train:
            if image_id not in train_descriptions:
                train_descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            train_descriptions[image_id].append(desc)
    return train_descriptions

### vocab (list) ###  
def make_vocab(train_descriptions):
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
    
    ### ixtoword, wordtoix (dict), vocab_size ###
    ixtoword = {}
    wordtoix = {}
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    vocab_size = len(ixtoword) + 1
    
    ### max_length (seq) ### 
    all_desc = list()
    for key in train_descriptions.keys():
        [all_desc.append(d) for d in train_descriptions[key]]
    lines = all_desc
    max_length = max(len(d.split()) for d in lines)
    
    return vocab, wordtoix, ixtoword, vocab_size, max_length 

### embedding_index (dict), embedding_matrix (numpy) ###
def make_text_embedding(vocab_size, wordtoix, base_path, txt_name='glove.6B.200d.txt', embedding_dim=200):
    embeddings_index = {} 
    f = open(os.path.join(base_path, txt_name), encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in wordtoix.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
