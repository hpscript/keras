from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

filepath_dict = {
	'yelp': 'data/yelp_labelled.txt',
	'amazon': 'data/amazon_cells_labelled.txt',
	'imdb': 'data/imdb_labelled.txt'
}

df_list = []
for source, filepath in filepath_dict.items():
	df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
	df['source'] = source
	df_list.append(df)

df = pd.concat(df_list)

df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(
	sentences, y, test_size=0.25, random_state=1000)


tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences_train)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 50
maxlen = 100

model = Sequential()
model.add(layers.Embedding(input_dim=vocab_size,
                            output_dim=embedding_dim,
                            input_length=maxlen))

model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
# print(model.summary())





 

