from arg.perspectives.collection_based_classifier import learn_lm, mention_num_classifier
from cache import load_from_pickle
from list_lib import lmap, foreach
from misc_lib import average
from models.classic.stopword import load_stopwords


def load_feature_and_split():
    print("Loading data")
    train_data = load_from_pickle("pc_train_features_binary")
    split = int(0.7 * len(train_data))

    train = train_data[:split]
    val = train_data[split:]
    return train, val


def remove_stopword_and_punct(stopwords, counter):
    punct = ".,'`\""
    keys = list(counter.keys())
    for key in keys:
        if key in stopwords or key in punct:
            counter.pop(key)


def test_generative_model():
    train, val = load_feature_and_split()
    print("Training lm")
    classifier = learn_lm(train)
    stopwords = load_stopwords()

    def fileter_fn(data_point):
        remove_stopword_and_punct(stopwords, data_point[0][0])

    foreach(fileter_fn, train)

    def is_correct(elem):
        x, y = elem
        x = x[0]
        return classifier.predict(x) == int(y)

    correctness = lmap(is_correct, val)

    print("val acc: ", average(correctness))


 #
def lm_contribution():
    train, val = load_feature_and_split()
    print("Training lm")
    stopwords = load_stopwords()

    def fileter_fn(data_point):
        remove_stopword_and_punct(stopwords, data_point[0][0])

    foreach(fileter_fn, train)
    classifier = learn_lm(train)

    for data_point in train:
        (tf, num), y = data_point

        contrib = classifier.counter_contribution(tf)
        print("{} {} {}".format(y, classifier.predict(tf), classifier.counter_odd(tf)))
        print("--------------")
        for t, score in contrib.most_common(20):
            print(t, score)


def mention_num_based():
    train, val = load_feature_and_split()
    mention_num_classifier(train, val)


if __name__ =="__main__" :
    mention_num_based()
