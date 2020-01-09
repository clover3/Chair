import os
from datetime import timedelta, date

from cpath import data_path


def lines_contain(lines, keywords):
    for line in lines:
        for word in keywords:
            if word in line:
                return True
    return False


def load_tweets():
    dir_path = os.path.join(data_path, "controversy", "tweet")

    def daterange(start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    start_date = date(2018, 7, 1)
    end_date = date(2019, 6, 30)
    tweets = []
    for single_date in daterange(start_date, end_date):
        day_str = single_date.strftime("%Y-%m-%d")

        filepath = os.path.join(dir_path, day_str)

        lines = open(filepath, "r").readlines()

        tweets.append((day_str, lines))
        print(day_str, len(lines))
        if lines_contain(lines, ["controversy", "controversial"]):
            print(day_str)
            break

if __name__ == "__main__":
    load_tweets()
