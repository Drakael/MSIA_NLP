import MSAI_NLP
import keras
print('Using Keras', keras.__version__)
import numpy as np
import matplotlib.pyplot as plt
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
from keras.layers import Dropout, Dense
from keras.models import Sequential

from keras.datasets import imdb


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


lexicon_size = 20000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=lexicon_size)

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

reviews = {}
titles = {}
for i, encoded_review in enumerate(train_data):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    reviews[i] = decoded_review
    titles[i] = decoded_review[:20]

for i, title in enumerate(reviews):
    print(reviews[title])
    print('---------')
    if i > 20:
        break

msai_nlp = MSAI_NLP.MSAI_NLP()

msai_nlp.feed_corpus(reviews)
print('train_data len', len(train_data))
print('train_labels len', len(train_labels))
print('reviews len', len(reviews))
print('tfs_mat len', len(msai_nlp.tfs_mat))

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(msai_nlp.lexicon_size,)))
model.add(Dropout(0.4))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.4))
model.add(layers.Dense(1, activation='sigmoid'))

print(model.summary())

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

np.random.seed(17)
rand_sample_index = np.random.randint(len(train_data))

print(train_labels[rand_sample_index])
y_train = np.asarray(train_labels).astype('float32')
print(y_train[rand_sample_index])
y_test = np.asarray(test_labels).astype('float32')

x_val = msai_nlp.tfs_mat[:10000]
partial_x_train = msai_nlp.tfs_mat[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=3,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
print(history_dict.keys())

acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# "bo" is for "blue dot"
plt.plot(epochs, acc, 'bo', label='Training accuracy')
# b is for "solid blue line"
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
