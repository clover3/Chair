
from collections import Counter
from data_generator.data_parser.tweet_reader import load_per_user, topics


def view_same_author():

    topic = topics[0]
    print(topic)

    user_tweets = load_per_user(topic)

    valid_group = dict()
    def remove_duplicate(texts):
        return list(set(texts))

    print("Total users : {}".format(len(user_tweets)))
    n_multi_tweet = 0
    n_multi_tweet_filter = 0
    histogram = Counter()
    for key in user_tweets.keys():
        texts = user_tweets[key]
        if len(texts) > 1 :
            n_multi_tweet += 1
            texts = remove_duplicate(texts)
            histogram[len(texts)] += 1
            if len(texts) > 1:
                n_multi_tweet_filter += 1
                #print(texts[0])
                #print(texts[1])
                valid_group[key] = texts


    for i in range(50):
        print("{} : {}".format(i,histogram[i]))

    print("#user with multi tweets : {}".format(n_multi_tweet))
    print("#user with multi tweets(filter) : {}".format(n_multi_tweet_filter))

    cnt =0
    for key in valid_group.keys():
        l = valid_group[key]
        print("------")
        print(l[0])
        print(l[1])

        cnt += 1
        if cnt > 50 :
            break


if __name__ == "__main__":
    view_same_author()


