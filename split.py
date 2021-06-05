from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd 
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

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

vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)

X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
score = classifier.score(X_test, y_test)

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
				optimizer='adam',
				metrics=['accuracy'])
# print(model.summary())
history = model.fit(X_train, y_train,
					epochs=100,
					verbose=False,
					validation_data=(X_test, y_test),
					batch_size=10)
# loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.savefig("img.png")

plot_history(history)

# print("Accuracy:", score)
# for source in df['source'].unique():
# 	df_source = df[df['source'] == source]
# 	sentences = df_source['sentence'].values
# 	y = df_source['label'].values

# 	sentences_train, sentences_test, y_train, y_test = train_test_split(
# 		sentences, y, test_size=0.25, random_state=1000)

# 	vectorizer = CountVectorizer()
# 	vectorizer.fit(sentences_train)
# 	X_train = vectorizer.transform(sentences_train)
# 	X_test = vectorizer.transform(sentences_test)

# 	classifier = LogisticRegression()
# 	classifier.fit(X_train, y_train)
# 	score = classifier.score(X_test, y_test)
# 	print('Accuracy for {} data: {:.4f}'.format(source, score))