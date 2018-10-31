import sklearn
import pandas as pd
#from nltk.corpus import stopwords


def bag_of_words(reviews, **kwargs):
    unigram_vect = sklearn.feature_extraction.text.CountVectorizer(
        analyzer="word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
#        stop_words=stopwords.words('english'),
        max_features=None)

    vect_train = unigram_vect.fit_transform(train)
    vect_test = unigram_vect.transform(test)

    if kwargs['transform'] == True:
        unigram_train = unigram_vect.transform(reviews)

        unigram_train = unigram_vect.fit_transform(reviews)
    else:
        unigram_train = unigram_vect.fit_transform(reviews)

    return pd.DataFrame(vect_fit.A, columns=unigram_vect.get_feature_names())
