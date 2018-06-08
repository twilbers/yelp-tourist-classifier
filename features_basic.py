import time


def cities_to_num(cities):
    cities_dict = {}
    results = []
    count = 0
    for city in cities:
        if city not in cities_dict:
            count += 1
            cities_dict[city] = count
            results.append(count)
        else:
            results.append(cities_dict[city])
    return results


def week_of_year(date):
    struct_time = time.strptime(date, "%Y-%m-%d")
    return struct_time.tm_yday // 7


def day_of_week(date):
    return time.strptime(date, "%Y-%m-%d").tm_wday


def city_mentioned(reviews, cities):
    cities_named = []
    for i in range(len(reviews)):
        if cities[i] in reviews[i] or cities[i].lower() in reviews[i]:
            cities_named.append(1)
        else:
            cities_named.append(0)
    return cities_named


def length(reviews):
    return [len(review) for review in reviews]
