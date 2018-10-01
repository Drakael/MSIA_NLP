import operator
import numpy as np


# fonction utile pour le tracing
def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


class MSAI_NLP():
    """NLP class
    """

    def __init__(self):
        self.corpus = {}
        self.texts = []
        self.titles = []
        self.corpus_counts = []
        self.total_counts = {}
        self.lexicon_size = None
        self.mean_counts = {}
        self.presence_counts = {}
        self.tfidfs = []
        self.tfidf_sorted = None
        self.lexicon_index = []
        self.word_to_id = {}

    def clean_text(self, text):
        text = text.lower()
        text = text.replace("'", "' ")
        text = text.replace(':', '')
        text = text.replace('"', '')
        text = text.replace(';', '')
        text = text.replace(',', '')
        text = text.replace('.', '')
        text = text.replace('-', ' ')
        # text = text.replace('')
        return text

    def split(self, text):
        text = self.clean_text(text)
        return text.split()

    def count_words(self, text):
        counts = {}
        words = self.split(text)
        for word in words:
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1
        return counts

    def get_corpus_count(self):
        self.corpus_counts = []
        for text in self.texts:
            counts = self.count_words(text)
            self.corpus_counts.append(counts)
        return self

    def get_total_counts(self):
        self.total_counts = {}
        for text_counts in self.corpus_counts:
            for word in text_counts:
                if word in self.total_counts:
                    self.total_counts[word] += text_counts[word]
                else:
                    self.total_counts[word] = text_counts[word]
        self.lexicon_size = len(self.total_counts)
        self.total_counts_sorted = sorted(self.total_counts.items(),
                                          key=operator.itemgetter(1))[::-1]
        return self

    def get_mean_counts(self):
        self.mean_counts = {}
        for word, count in self.total_counts_sorted:
            self.mean_counts[word] = count / len(self.texts)
        return self

    def get_presence_counts(self):
        self.presence_counts = {}
        for text in self.texts:
            words = self.split(text)
            for word in words:
                if word in self.presence_counts:
                    self.presence_counts[word] += 1
                else:
                    self.presence_counts[word] = 1
        return self

    def get_tfidf(self):
        self.tfidfs = []
        for text_counts in self.corpus_counts:
            tfidf = {}
            for word in text_counts:
                # tfidf[word] = fable_counts[word] / total_counts[word]
                tfidf[word] = np.log(len(self.texts) / self.presence_counts[word])
            self.tfidfs.append(tfidf)
        return self

    def build_lexicon_index(self):
        self.lexicon_index = [(i, word) for i, word in enumerate(self.total_counts)]
        self.word_to_id = {word: i for (i, word) in self.lexicon_index}
        return self

    def build_df_vec(self):
        self.df_vec = np.zeros((1, len(self.lexicon_index)))
        for i, word in self.lexicon_index:
            self.df_vec[0, i] = self.total_counts[word]
        return self

    def build_tfs_mat(self):
        self.tfs_mat = np.zeros((len(self.texts), len(self.lexicon_index)))
        for j, tfidf in enumerate(self.tfidfs):
            for tword in tfidf:
                id_col = self.word_to_id[tword]
                self.tfs_mat[j, id_col] = tfidf[tword]
        return self

    def text2vec(self, text, word_to_id=None):
        if word_to_id is None:
            word_to_id = self.word_to_id
        split = self.split(text)
        vec = np.zeros((1, len(word_to_id)))
        for word in split:
            if word in word_to_id:
                id_col = word_to_id[word]
                vec[0, id_col] = 1
            else:
                print('Word not in lexicon : ', word)
        return vec

    def texts2matrix(self, texts, lexicon_size=None):
        if texts and type(texts) is not list:
            raise ValueError('texts must be a list')
        if not all(isinstance(text, str) for text in texts):
            raise ValueError('texts must be a list of strings')
        if lexicon_size is None:
            lexicon_size = self.lexicon_size
        sequences = [self.text2vec(text) for text in texts]
        # Create an all-zero matrix of shape (len(sequences), lexicon_size)
        results = np.zeros((len(sequences), lexicon_size))
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1
        return results

    def scalar_product(self, v, w):
        return np.sum(v * w)

    def vec_length(self, v):
        return np.sum(v**2)**0.5

    def cosine_similarity(self, v, w):
        len_product = self.vec_length(v) * self.vec_length(w)
        return self.scalar_product(v, w) / len_product

    def search(self, search):
        vec_search = self.text2vec(search, self.word_to_id)
        self.answers = {}
        for i, tf_vec in enumerate(self.tfs_mat):
            cos_sim = self.cosine_similarity(vec_search, tf_vec)
            self.answers[self.titles[i]] = round(1000 * cos_sim, 2)

        self.answers_sorted = sorted(self.answers.items(),
                                     key=operator.itemgetter(1))[::-1]
        print('answers_sorted', self.answers_sorted)

    def feed_corpus(self, corpus):
        if corpus and type(corpus) is dict:
            if not all(isinstance(corpus[text], str) for text in corpus):
                raise ValueError('corpus must be a dict or list of strings')
            for title in corpus:
                self.texts.append(corpus[title])
                self.titles.append(title)
                self.corpus = corpus
        else:
            if corpus and type(corpus) is not list:
                raise ValueError('corpus must be a dict or list')
            if not all(isinstance(text, str) for text in corpus):
                raise ValueError('corpus must be a dict or list of strings')
            for i, text in enumerate(corpus):
                self.texts.append(text)
                self.titles.append(i)
            self.corpus = zip(self.titles, self.texts)
        self.get_corpus_count().get_total_counts().get_presence_counts()
        self.get_tfidf().build_lexicon_index().build_df_vec().build_tfs_mat()

    def sort_tfidf(self):
        self.tfidf_sorted = sorted(self.tfidfs[0].items(),
                                   key=operator.itemgetter(1))[::-1]
