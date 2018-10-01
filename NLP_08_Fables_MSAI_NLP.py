import os
import MSAI_NLP
import operator

rootdir = 'Fables La Fontaine'

fables = []
titres = []
corpus = {}
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file[-4:] == '.txt':
            print(file)
            f = open(rootdir + '/' + file, 'r', encoding='utf8')
            texte = f.read()
            fables.append(texte)
            titre = file[:-4]
            titres.append(titre)
            corpus[titre] = texte

print(len(fables))

msai_nlp = MSAI_NLP.MSAI_NLP()

msai_nlp.feed_corpus(corpus)

print('lexicon_index len', len(msai_nlp.lexicon_index))
search = 'raison du plus fort'
print('search for', search)

# print('tfidfs', msai_nlp.tfidfs)
tfidf_sorted = sorted(msai_nlp.tfidfs[0].items(),
                      key=operator.itemgetter(1))[::-1]
print(tfidf_sorted)

msai_nlp.search(search)
