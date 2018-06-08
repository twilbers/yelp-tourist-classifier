import sklearn
import pandas as pd


def bag_of_words(reviews):
    unigram_vect = sklearn.feature_extraction.text.CountVectorizer(
                                analyzer ="word",
                                tokenizer =None,
                                preprocessor =None,
                                stop_words =stopwords.words('english'),
                                max_features= None)

    vect_fit = unigram_vect.fit_transform(reviews)
    return pd.DataFrame(vect_fit.A, columns=unigram_vect.get_feature_names())
