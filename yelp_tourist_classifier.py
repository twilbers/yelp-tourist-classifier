import features_basic
from features_pos import pos_counter, get_pos_pickle, pos_to_dict

import numpy as np
import pandas as pd

from scipy.sparse.csr import csr_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC


def add_basic_features():
    """
    add_basic_features(data) takes a datafram with
    review labels as first column as an input and
    tputs an extended dataframe with features based
    on the Yelp reviews meta data.
    """
    data = pd.read_csv('data/all_reviews_cleaned.csv')

    # Add features
    data = features_basic.review_len(data, 'review_text')

    # week of year
    data = features_basic.week_of_year(data, 'review_date')

    # day of week
    data = features_basic.day_of_week(data, 'review_date')

    # city mentioned
    data = features_basic.city_mentioned(
        data, 'business_city', 'review_text')

    return data


def add_pos_features(data):
    """
    add_pos_features(data) takes a dataframe
    as an input and outputs an exteded dataframe
    with part of speech features added using SPOT.
    """
    reviews_tagged = get_pos_pickle(
        'dump/scraped_reviews_tagged.p')

    pos_list = pos_to_dict(reviews_tagged)

    pos_keep = [
        '#', 'RBS', 'RBR', 'VBZ', 'VBD',
        'VBP', 'JJR', 'LS', 'MD', 'VBN',
        'EX', '$']

    data = data.join(
        pd.DataFrame.from_dict(pos_list)[pos_keep],
        on=None, how='left', lsuffix='', rsuffix='', sort=False)

    return data


def saliance(words, local_words, remote_words, theta=.50):
    """ saliance(data) takes a dataframe and returns a list of variables to keep
    that meet a salience theta
    """
    keep_words = []
    for i in range(words.shape[1]):
        normalizer = words[:, i].sum()
        l_prob_sum = local_words[:, i].sum() / normalizer
        r_prob_sum = remote_words[:, i].sum() / normalizer

        min_ = min(r_prob_sum, l_prob_sum)
        max_ = max(r_prob_sum, l_prob_sum)
        if max_ != 0:
                salience = (1 - (min_ / max_))
        else:
                salience = 0
        if salience > theta:
            keep_words.append(i)
    return keep_words


def sampe_data(data):
    state_min = min(data.query('business_state != "NJ"').groupby(
        'business_state').agg('count').iloc[:, 0])

    sample_ny = data.query('business_state == "NY"').sample(n=state_min)
    sample_nv = data.query('business_state == "NV"').sample(n=state_min)
    sample_ca = data.query('business_state == "CA"').sample(n=state_min)
    sample_fl = data.query('business_state == "FL"').sample(n=state_min)
    sample_il = data.query('business_state == "IL"').sample(n=state_min)

    sample = pd.concat(
        [sample_ny, sample_nv, sample_ca,
            sample_fl, sample_il]).reset_index(drop=True)

    sample_min = min(sample.groupby('label_').agg('count').iloc[:, 0])

    local_sample = sample.query('label_ == "local"').sample(n=sample_min)
    remote_sample = sample.query('label_ == "remote"').sample(n=sample_min)

    return pd.concat([local_sample, remote_sample]).reset_index(drop=True)


yelp_df = add_basic_features()

yelp_df = yelp_df.rename(
    columns={'cool': 'cool_', 'label': 'label_',
             'funny': 'funny_', 'useful': 'useful_'})

yelp_df = add_pos_features(yelp_df)

le_zip = LabelEncoder()
yelp_df[['business_zip']] = le_zip.fit_transform(yelp_df['business_zip'])


X = yelp_df.drop([
    'business_city', 'business_state', 'business_name',
    'reviewer_location', 'business_url', 'review_date',
    'reviewer_id', 'funny_', 'cool_'], axis=1)

print("Model feature space includes:", ', '.join(X.columns))

y = yelp_df[['label_']].astype('category')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.75, test_size=0.25)

unigram_vect = CountVectorizer(
    analyzer="word",
    tokenizer=None,
    ngram_range=(1, 2),
    preprocessor=None,
    stop_words=None,
    max_features=10000)

unigram_fit = unigram_vect.fit_transform(X_train['review_text'])
unigram_transform = unigram_vect.transform(X_test['review_text'])

local_words = unigram_vect.transform(
    X_train.query('label_ == "local"')['review_text'])
remote_words = unigram_vect.transform(
    X_train.query('label_ == "remote"')['review_text'])
keep_index = saliance(unigram_fit, local_words, remote_words, theta=.25)

unigram_transform = csr_matrix(unigram_transform[:, keep_index])
unigram_fit = csr_matrix(unigram_fit[:, keep_index])
keep_words = np.array(unigram_vect.get_feature_names())[keep_index]

unigram_train = pd.DataFrame(unigram_fit.A, columns=keep_words)

unigram_test = pd.DataFrame(unigram_transform.A, columns=keep_words)

print(unigram_train.shape[1], " n-grams in model")
print(unigram_train.columns)


X_train = X_train.drop(['review_text', 'label_'], axis='columns')
X_train = X_train.join(
    unigram_train, on=None, how='left', lsuffix='',
    rsuffix='', sort=False).fillna(0)

X_test = X_test.drop(['review_text', 'label_'], axis='columns')
X_test = X_test.join(
    unigram_test, on=None, how='left', lsuffix='',
    rsuffix='', sort=False).fillna(0)

X_std = StandardScaler()
X_train = X_std.fit_transform(X_train)
X_test = X_std.transform(X_test)


gnb = GaussianNB()
gnb.fit(X_train, y_train.values.ravel())

# logistic_lib = LogisticRegression(solver='liblinear')
# logistic_lbf = LogisticRegression(solver='lbfgs')
# logistic_newton = LogisticRegression(solver='newton-cg')

logistic = LogisticRegression(solver='liblinear')
logistic.fit(X_train, y_train.values.ravel())


score_lr = logistic.score(X_test, y_test)
score_nb = gnb.score(X_test, y_test)

print(
    'Logistical regression:', score_lr, '\n',
    'Naive Bayes :', score_nb)
