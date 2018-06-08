from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize
from os import environ
import pickle

def review_tokenize(reviews):
    return [word_tokenize(review) for review in reviews]


def review_tager(tokenized_reviews):
    # java_path = "C:/Program Files (x86)/Java/jre1.8.0_31/bin/java.exe"
    java_path = environ['JAVAHOME']
    st_model_path = 'tools/SPOST/models/english-bidirectional-distsim.tagger'
    st = StanfordPOSTagger(st_model_path,
                           'tools/SPOST/stanford-postagger.jar')
    results = []
    errors = []
    count = 0
    for review in tokenized_reviews:
        if count % 50 == 0:
            print(count)
        try:
            results.append(st.tag(review))
            count += 1
        except:
            errors.append(count)
            results.append(review)
            count += 1
    print('errors for the following indexes\n', errors)
    return results  # [st.tag(review) for review in tokenized_reviews]


def get_pos_pickle():
    with open('dump/reviews_tagged', 'rb') as f_reviews_tagged:
        return pickle.load(f_reviews_tagged)


class pos_counter():

    adverbs = [u'RB', u'RBR', u'RBS', u'RBS\r', u'RB\r', u'RBR\r']
    simple_past = [u'VBD', u'VBD\r']
    simple_present = [u'VBP', u'VPZ', u'VBP\r', u'VPZ\r']
    past_participle = [u'VBN', u'VBN\r']
    modal = [u'MD', u'MD\r']
    pn = [u'NNP', u'NNPS', u'NNP\r', u'NNPS\r']
    prep = [u'IN', u'IN\r']
    nn = [u'NN', u'NN\r']
    adj = [u'JJ', u'JJ\r']
    dt = [u'DT', u'DT\r']

    def count_pos(tagged_reviews, pos_list):
        count = 0
        for review in tagged_reviews:
            try:
                if review[1] in pos_list:
                    count += 1
            except:
                pass
        return float(count)

def superlative_count():
    pass


def capital_words():
    pass


def exclamitory_count():
    pass


def plural_pronouns():
    pass