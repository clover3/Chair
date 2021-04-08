from typing import List

from sklearn.svm import LinearSVC

from models.baselines import svm


class SVMWrap:
    def __init__(self, train_x, train_y):
        feature_extractor = svm.NGramFeature(False, 4)
        self.feature_extractor = feature_extractor
        # _, _, argu_ana_dev_x, argu_ana_dev_y = get_data()
        X_train_counts = feature_extractor.fit_transform(train_x)
        svclassifier = LinearSVC()
        svclassifier.fit(X_train_counts, train_y)
        self.svclassifier = svclassifier

    def predict(self, test_x: List[str]) -> List[float]:
        x_test_count = self.feature_extractor.transform(test_x)
        test_pred = self.svclassifier.decision_function(x_test_count)
        return test_pred

