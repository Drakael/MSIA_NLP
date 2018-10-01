import MSAI_NLP

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
for i, encoded_review in enumerate(train_data):
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])
    reviews[decoded_review[:20]] = decoded_review

for i, title in enumerate(reviews):
    print(reviews[title])
    print('---------')
    if i > 20:
        break

msai_nlp = MSAI_NLP.MSAI_NLP()

msai_nlp.feed_corpus(reviews)

print('lexicon_index len', len(msai_nlp.lexicon_index))
search = 'george lucas'
print('search for', search)

msai_nlp.search(search)

count = 0
for title, score in msai_nlp.answers_sorted:
    if score == 0:
        break
    print(reviews[title])
    print('------------------')
    count += 1

print('nb results', count)
