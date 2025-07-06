import pandas as pd
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

#
# load data
#

data_raw = pd.read_csv("./reddit_data.csv")
df = pd.read_csv("reddit_data.csv").head(100000)

#
# preprocessing (vocab, vocab_probs)
#

def normalize(lst):
    s = sum(lst)
    normed = [itm/s for itm in lst]
    # pad last value with what ever difference neeeded to make sum to exactly 1
    normed[-1] = (normed[-1] + (1-sum(normed)))
    return normed

vocab_counts = df["subreddit"].value_counts()
tmp_vocab = list(vocab_counts.keys())
total_counts = sum(vocab_counts.values)

inv_prob = [total_counts/vocab_counts[sub] for sub in tmp_vocab]
vocab = ["Unseen-Sub"] + tmp_vocab
tmp_vocab_probs = normalize(inv_prob)

vocab_probs = [1-sum(tmp_vocab_probs)] + tmp_vocab_probs
print("Vocab size = " + str(len(vocab)))

#
# chunking (dictionary: {username: [sub_usr, ...], ...})
#

def remove_repeating_subs(df):
    cache_data = {}
    prev_usr = None
    past_sub = None
    for comment_data in df.itertuples():
        current_usr = comment_data[1]
        if current_usr != prev_usr: 
            if prev_usr != None and prev_usr not in cache_data.keys():
                cache_data[prev_usr] = usr_sub_seq
            usr_sub_seq = [comment_data[2]]
            past_sub = comment_data[2]
        else: 
            if comment_data[2] != past_sub: 
                usr_sub_seq.append(comment_data[2])
                past_sub = comment_data[2]
        prev_usr = current_usr 
    return cache_data

sequence_chunk_size = 15
pp_user_data = remove_repeating_subs(df)

def chunks(l, n=15): 
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

# vocab, vocab_probs -> train_seqs (chnk_seq, label, len(chnk_seq))
def build_training_sequences(usr_data):
    train_seqs = []
    for _, usr_sub_seq in usr_data.items():
        comment_chunks = chunks(usr_sub_seq, sequence_chunk_size)
        for chnk in comment_chunks:
            filtered_subs = [vocab.index(sub) for sub in chnk] # vocab to ids
            if filtered_subs:
                filter_probs = normalize([vocab_probs[sub_indx] for sub_indx in filtered_subs]) # softmax
                label = np.random.choice(filtered_subs, 1, p=filter_probs)[0]
                chnk_seq = [vocab.index(sub) for sub in chnk if sub in vocab and vocab.index(sub) != label] 
                train_seqs.append([chnk_seq, label, len(chnk_seq)]) 
    return train_seqs

train_data = build_training_sequences(pp_user_data)
seqs, lbls, lngths = zip(*train_data)
train_df = pd.DataFrame({'sub_seqs':seqs,
                         'sub_label':lbls,
                         'seq_length':lngths})
# train_df.head()

#
# train  
#
 
def train_model(train, test, vocab_size, n_epoch=5, n_units=128, dropout=0.6, learning_rate=0.0001):
    trainX = train['sub_seqs']
    trainY = train['sub_label']
    testX =  test['sub_seqs']
    testY =  test['sub_label']

    trainX = pad_sequences(trainX, maxlen=sequence_chunk_size, value=0., padding='post')
    testX = pad_sequences(testX, maxlen=sequence_chunk_size, value=0., padding='post')

    trainY = to_categorical(trainY, num_classes=vocab_size)
    testY = to_categorical(testY, num_classes=vocab_size)
 
    inputs = Input(shape=(sequence_chunk_size,), dtype='int32')
    x = Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(inputs)
    x = LSTM(n_units, dropout=dropout)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=512, epochs=n_epoch, verbose=2)
    return model

split_perc = 0.8
train_len, test_len = np.floor(len(train_df) * split_perc), np.floor(len(train_df) * (1 - split_perc))
train, test = train_df.loc[:train_len - 1], train_df.loc[train_len:train_len + test_len]
model = train_model(train, test, len(vocab))

# #### analysis ####

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# em_layer = model.get_layer('Embedding_layer')
# em_weights = em_layer.get_weights()[0]

# tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
# lowDWeights = tsne.fit_transform(em_weights)

# #### ####

# from bokeh.plotting import figure, show, output_notebook,output_file
# from bokeh.models import ColumnDataSource, LabelSet

# #control the number of labelled subreddits to display
# sparse_labels = [lbl if random.random() <=0.01 else '' for lbl in vocab]
# source = ColumnDataSource({'x':lowDWeights[:,0],'y':lowDWeights[:,1],'labels':sparse_labels})


# TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"

# p = figure(tools=TOOLS)

# p.scatter("x", "y", radius=0.1, fill_alpha=0.6,
#           line_color=None,source=source)

# labels = LabelSet(x="x", y="y", text="labels", y_offset=8,
#                   text_font_size="10pt", text_color="#555555", text_align='center',
#                  source=source)
# p.add_layout(labels)

# output_notebook()
# show(p)
