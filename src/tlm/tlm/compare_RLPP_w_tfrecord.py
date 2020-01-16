import os
import pickle
import random

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from cpath import output_path
from misc_lib import average
from tf_util.enum_features import load_record_v2
from tlm.estimator_prediction_viewer import flatten_batches, EstimatorPredictionViewerGosford
from visualize.html_visual import HtmlVisualizer, Cell


def discretize(p):
    return int(p * 10 + 0.5)*10

def useful(p1, p2):
    if p1 < p2:
        p1 = p2

    return p2 / p1
    p1 = discretize(p1)
    p2 = discretize(p2)



def do_verfiy():
    pred_file_name = "1.pickle"
    record_file_name = "C:\\work\\Code\\Chair\\output\\1"
    p = os.path.join(output_path, pred_file_name)
    data = pickle.load(open(p, "rb"))
    data = flatten_batches(data)
    itr1 = load_record_v2(record_file_name)
    itr2 = data["prob1"]
    print("itr2 len", len(itr2))
    cnt = 0
    for _ in itr1:
        cnt += 1
    print(cnt)
    for _ in itr2:
        cnt -= 1
    print(cnt)
    print(cnt)

def do():
    pred_file_name = "RLPP_0.pickle"
    pred_file_name = "ukp_rel.pickle"
    record_file_name = "C:\\work\\Code\\Chair\\output\\unmasked_pair_x3_0"
    record_file_name = "C:\\work\\Code\\Chair\\output\\tf_enc"
    todo = [
        ("RLPP_0.pickle", "C:\\work\\Code\\Chair\\output\\unmasked_pair_x3_0", "RLPP_wiki.html"),
        ("ukp_rel.pickle", "C:\\work\\Code\\Chair\\output\\tf_enc", "RLPP_ukp.html" )
    ]
    x = []
    y = []
    for pred_file_name, record_file_name, out_name in todo:
        viewer = EstimatorPredictionViewerGosford(pred_file_name)
        html = HtmlVisualizer(out_name)
        itr1 = load_record_v2(record_file_name)
        itr2 = viewer.__iter__()
        cnt = 0
        for features, entry in zip(itr1, itr2):
            cnt += 1
            if cnt > 200:
                break
            input_ids1 = entry.get_tokens("input_ids")
            prob1 = entry.get_vector("prob1")
            prob2 = entry.get_vector("prob2")

            cells = viewer.cells_from_tokens(input_ids1)
            p1_l = []
            p2_l = []
            useful_l = []

            row1 = []
            row2 = []
            row3 = []
            row4 = []
            for j, cell in enumerate(cells):
                p1 = float(prob1[j])
                p2 = float(prob2[j])
                x.append([p1])
                y.append(p2)
                u = useful(p1, p2)
                score = (1-u) * 100
                cell.highlight_score = score
                row1.append(cell)
                row2.append(Cell(p1, score))
                row3.append(Cell(p2, score))
                row4.append(Cell(u, score))

                p1_l.append(p1)
                p2_l.append(p2)
                useful_l.append(u)
                if len(row1) > 20:
                    rows = [row1, row2, row3, row4]
                    row1 = []
                    row2 = []
                    row3 = []
                    row4 = []
                    html.write_table(rows)

            html.write_paragraph("p1: {}".format(average(p1_l)))
            html.write_paragraph("p2: {}".format(average(p2_l)))
            html.write_paragraph("useful: {}".format(average(useful_l)))

            if average(useful_l) < 0.4:
                html.write_headline("Low Score")

        l = list(zip(x,y))
        random.shuffle(l)
        l = l[:1000]
        x,y = zip(*l)
        lin = LinearRegression()
        lin.fit(x, y)

        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(x)
        poly.fit(X_poly , y)
        lin2 = LinearRegression()
        lin2.fit(X_poly, y)
        plt.scatter(x, y, color='blue')

        plt.plot(x, lin2.predict(poly.fit_transform(x)), color='red')
        plt.title('Polynomial Regression')

        plt.show()


if __name__ == "__main__":
    do_verfiy()
