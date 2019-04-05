from data_generator.argmining import ukp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from task.metrics import eval_3label


class ArgExperiment:
    def __init__(self):
        pass


    def train_lr(self):
        topic = ukp.all_topics[0]

        data_loader = ukp.DataLoader(topic)
        idx_for = data_loader.labels.index("Argument_for")
        idx_against = data_loader.labels.index("Argument_against")

        train_data = data_loader.get_train_data()
        dev_data = data_loader.get_dev_data()

        train_X, train_y = zip(*train_data)
        dev_X, dev_y = zip(*dev_data)
        feature = CountVectorizer()
        train_X_v = feature.fit_transform(train_X)
        dev_X_v = feature.transform(dev_X)

        lr = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
        lr.fit(train_X_v, train_y)

        train_pred = lr.predict(train_X_v)
        dev_pred = lr.predict(dev_X_v)


        def print_eval(pred_y, gold_y):
            all_result = eval_3label(pred_y, gold_y)
            for_result = all_result[idx_for]
            against_result = all_result[idx_against]
            f1 = sum([result['f1'] for result in all_result]) / 3
            print("F1", f1)
            print("P_arg+", for_result['precision'])
            print("R_arg+", for_result['recall'])
            print("P_arg-", against_result['precision'])
            print("R_arg-", against_result['recall'])

        print("Train")
        print_eval(train_pred, train_y)
        print("Dev")
        print_eval(dev_pred, dev_y)
