import pandas as pd
import numpy as np
import time


def reviewer_state(df, reviewer_location):
    return df.assign(reviewer_state=df[reviewer_location].astype(str).apply(
        lambda x: x[-2:]))


def week_of_year(df, date_column):
    return df.assign(week_of_year=df[date_column].apply(
        lambda x: time.strptime(x, "%m/%d/%Y").tm_yday // 7))


def day_of_week(df, date_column):
    return df.assign(day_of_week=df[date_column].apply(
        lambda x: time.strptime(x, "%m/%d/%Y").tm_wday))


def city_mentioned(df, text_field1, text_field2):
    vecIn = np.vectorize(lambda a, b: a.lower() in b.lower())
    return df.assign(city_mentioned=np.where(
        vecIn(df[text_field1].values, df[text_field2].values), 1, 0))


def review_len(df, text_column):
    return df.assign(
        review_length=df[text_column].apply(
            lambda x: len(x)))


def elite_status(df, text_column):
    pass
