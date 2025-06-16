from tensorflow.keras import Input, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, Activation, Reshape, Dropout, add

def build_model(max_length, vocab_size, embedding_matrix, embedding_dim=200, INCEPTION_VEC=2048):
    ### image ###
    inputs1 = Input(shape=(INCEPTION_VEC,), name="image_input")
    fe1 = Dropout(0.5, name="image_dropout")(inputs1)
    fe2 = Dense(256, activation='relu', name="image_dense")(fe1)

    ### text ###
    inputs2 = Input(shape=(max_length,), name="text_input")
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True, name="text_embedding")(inputs2)
    se2 = Dropout(0.5, name="text_dropout")(se1)
    se3 = LSTM(256, name="text_LSTM")(se2)

    ### decoder ###
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    model.get_layer("text_embedding").set_weights([embedding_matrix])
    model.get_layer("text_embedding").trainable = False  
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

