import csv


class remote_import:

    reviews = []
    cities = []
    bstars = []
    trendy = []
    touristy = []
    hipster = []
    funny = []
    useful = []
    cool = []
    rstars = []
    dates = []
    users = []
    business = []

    with open('data/remote_reviews.csv', 'r') as f_remote:
        reader = csv.reader(f_remote, delimiter=';')
        fields = [field for field in next(reader)]

        for row in reader:
            users.append(row[0])
            business.append(row[1])
            reviews.append(row[2].replace('\,', ',').replace("'", "\'"))
            cities.append(row[3])
            bstars.append(row[5])
            trendy.append(row[8])
            touristy.append(row[9])
            hipster.append(row[10])
            funny.append(row[11])
            useful.append(row[12])
            cool.append(row[13])
            rstars.append(row[14])
            dates.append(row[15])


class local_import:

    with open('data/local_reviews.csv', 'r') as f_local:
        reader = csv.reader(f_local, delimiter=';')
        fields = [field for field in next(reader)]

        reviews = []
        cities = []
        bstars = []
        trendy = []
        touristy = []
        hipster = []
        funny = []
        useful = []
        cool = []
        rstars = []
        dates = []
        users = []
        business = []

        for i, row in enumerate(reader):
            if i == 1420:  # stop when length is same as remote_reviews
                break
            users.append(row[0])
            business.append(row[1])
            reviews.append(row[2].replace('\,', ',').replace("'", "\'"))
            cities.append(row[3])
            bstars.append(row[5])
            trendy.append(row[8])
            touristy.append(row[9])
            hipster.append(row[10])
            funny.append(row[11])
            useful.append(row[12])
            cool.append(row[13])
            rstars.append(row[14])
            dates.append(row[15])
