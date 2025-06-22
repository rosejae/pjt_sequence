import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, SimpleRNN
from tensorflow.keras.models import Model, load_model
from sklearn.decomposition import PCA
import seaborn as sns
import pdb, os

def build_model(n_songs, n_users, e_units=5, r_units=20):
    song_input = Input(shape=[1], name="Song-Input")
    song_embedding = Embedding(n_songs+1, e_units, name="Song-Embedding")(song_input)
    # song_embedding1 = SimpleRNN(r_units, name="Song-Embedding1")(song_embedding)
    song_vec = Flatten(name="Flatten-Songs")(song_embedding)

    user_input = Input(shape=[1], name="User-Input")
    user_embedding = Embedding(n_users+1, e_units, name="User-Embedding")(user_input)
    user_vec = Flatten(name="Flatten-Users")(user_embedding)

    prod = Dot(name="Dot-Product", axes=1)([song_vec, user_vec])
    model = Model([user_input, song_input], prod)
    return model

dataset = pd.read_csv("./song_data.csv")
train, test = train_test_split(dataset, test_size=0.2, random_state=42)

n_songs = len(dataset.song_id.unique())
n_users = len(dataset.user_id.unique())

save_path = "music_model.h5"
if os.path.exists(save_path):
    model = load_model("music_model.h5")
    model.summary()
else: 
    model = build_model(n_songs, n_users)
    model.compile("adam", "mean_squared_error")
    history = model.fit([train.user_id, train.song_id], train.rating, epochs=10, verbose=1)

# Extract embeddings
song_em = model.get_layer("Song-Embedding")
song_em_weights = song_em.get_weights()[0]

pca = PCA(n_components=2)
pca_result = pca.fit_transform(song_em_weights)
sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1])

song_data = np.array(list(set(dataset.song_id)))
user = np.array([1 for i in range(len(song_data))])

predictions = model.predict([user, song_data])
predictions = np.array([a[0] for a in predictions])

recommended_song_ids = (-predictions).argsort()[:5]

print(recommended_song_ids)
print(predictions[recommended_song_ids]) # 이 확률을 바탕으로 모델은 아이디어를 내놓음