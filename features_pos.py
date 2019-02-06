from nltk.tag.stanford import StanfordPOSTagger
from nltk import word_tokenize
import pickle


def review_tokenize(reviews):
    return map(lambda review: word_tokenize(review), reviews)


def review_tager(tokenized_reviews):
    st_model_path = r'SPOST/models/english-bidirectional-distsim.tagger'
    st = StanfordPOSTagger(st_model_path,
                           r'SPOST/stanford-postagger.jar')
    results = []
    errors = []
    count = 0

    for review in tokenized_reviews:
        try:
            results.append(st.tag(review))
            count += 1
        except:
            print(count)
            errors.append(count)
            results.append(review)
            count += 1
    print('errors for the following indexes\n', errors)
    return results


def get_pos_pickle(file_location):
    with open(file_location, 'rb') as f_reviews_tagged:
        return pickle.load(f_reviews_tagged)


def pos_to_dict(tagged_reviews):
    tags = []
    pos_list = []
    for review in tagged_reviews:
        pos_dict = {}
        for item in review:
            if type(item) == tuple:
                if item[1] in pos_dict:
                    pos_dict[item[1]] += 1.0
                else:
                    pos_dict[item[1]] = 1.0
                if not item[1] in tags:
                    tags.append(item[1])
        pos_list.append(pos_dict)
    return pos_list

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