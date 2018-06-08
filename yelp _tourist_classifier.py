import pandas as pd
from review_import import remote_import, local_import
import features_basic
import features_pos
from  features_pos import pos_counter

import sklearn.model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


def get_dataframe():

    remote_reviews = remote_import.reviews
    local_reviews = local_import.reviews
    review_labels = (
        ['local'] * len(local_reviews)) + (['remote'] * len(remote_reviews))
    return pd.DataFrame(review_labels, columns=["review_labels"])


def add_basic_features(data):

    reviews = local_import.reviews + remote_import.reviews

    # city
    review_cities = local_import.cities + remote_import.cities
    data["review_cities"] = features_basic.cities_to_num(review_cities)

    # date
    review_dates = local_import.dates + remote_import.dates
    data["review_dates"] = [date for date in review_dates]

    # votes
    review_bstars = local_import.bstars + remote_import.bstars
    data["review_bstars"] = [int(i) for i in review_bstars]
    review_funny = local_import.funny + remote_import.funny
    data["review_funny"] = [int(i) for i in review_funny]
    review_useful = local_import.useful + remote_import.useful
    data["review_useful"] = [int(i) for i in review_useful]
    review_cool = local_import.cool + remote_import.cool
    data["review_cool"] = [int(i) for i in review_cool]
    review_rstars = local_import.rstars + remote_import.rstars
    data["review_rstars"] = [int(i) for i in review_rstars]

    # city mentioned in review
    data["city_mentioned"] = features_basic.city_mentioned(reviews, review_cities)

    # week of year
    review_dates = local_import.dates + remote_import.dates
    data["review_dates"] = [features_basic.week_of_year(date)
                            for date in review_dates]

    # day of week
    review_dates = local_import.dates + remote_import.dates
    data["review_dates"] = [features_basic.day_of_week(date)
                            for date in review_dates]

    # review length
    data["review_length"] = features_basic.length(reviews)
    return data


def add_pos_features(data):
    # reviews = local_import.reviews + remote_import.reviews
    # reviews_tokenized = review_tokenize(reviews)
    # reviews_tagged = features_pos.review_tokenize(reviews)
    reviews_tagged = features_pos.get_pos_pickle()
    data["adv count"] = [pos_counter.count_pos(review, pos_counter.adverbs)
                        for review in reviews_tagged]
    data["past prog"] = [pos_counter.count_pos(review, pos_counter.past_participle)
                        for review in reviews_tagged]
    data["simple future"] = [pos_counter.count_pos(review, pos_counter.modal)
                            for review in reviews_tagged]
    data["simple past"] = [pos_counter.count_pos(review, pos_counter.simple_past)
                            for review in reviews_tagged]
    data["simple present"] = [pos_counter.count_pos(review, pos_counter.simple_present)
                            for review in reviews_tagged]
    data['porper name'] = [pos_counter.count_pos(review, pos_counter.pn)
                            for review in reviews_tagged]
    data['prep count'] = [pos_counter.count_pos(review, pos_counter.prep)
                            for review in reviews_tagged]
    data['nn count'] = [pos_counter.count_pos(review, pos_counter.nn)
                            for review in reviews_tagged]
    data['adj count'] = [pos_counter.count_pos(review, pos_counter.adj)
                            for review in reviews_tagged]
    data['det count'] = [pos_counter.count_pos(review, pos_counter.dt)
                            for review in reviews_tagged]
    return data


def saliance(data):
    """ saliance(data) takes a dataframe and drops variables
    that do not meet a salience theta
    """
    theta = .75
    pass


data = get_dataframe()
data = add_basic_features(data)
data = add_pos_features(data)

X = data[data.columns[1:]].values
y = data["review_labels"].values

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
   X, y, train_size=0.75, test_size=0.25, random_state=333)

gnb = GaussianNB()
gnb.fit(X_train, y_train)

logistic = sklearn.linear_model.LogisticRegression()
logistic.fit(X_train, y_train)

score_lr = logistic.score(X_test, y_test)
score_nb = gnb.score(X_test, y_test)

print('Logistical regression:', score_lr, 'Naive bayes: ', score_nb)
