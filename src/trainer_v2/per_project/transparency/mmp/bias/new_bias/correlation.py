from tab_print import print_table
from trainer_v2.per_project.transparency.misc_common import load_tsv
from scipy.stats import pearsonr


def get_term_scores():
    years1 = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
              2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    relevance = [0.15991954505443573, 0.17009232938289642, 0.16455566883087158, 0.19133111834526062,
                 0.18887905776500702, 0.18211613595485687, 0.12253455817699432, 0.12562014162540436, 0.1550411880016327,
                 0.11106028407812119, 0.1153125986456871, 0.05630696937441826, 0.039196114987134933,
                 0.029429536312818527, 0.02278118208050728, 0.004900562111288309, 0.0026514031924307346,
                 0.0014096475206315517, 0.00046073851990513504, 0.14225299656391144, 0.13050927221775055,
                 0.15954212844371796, 0.030796818435192108, 0.045765507966279984, 0.06430057436227798]
    return relevance



def main():
    table_scores = get_term_scores()
    table = []
    for method in ["ce", "splade", "contriever", "contriever-msmarco", "tas_b"]:
        score_log_path = f"output/mmp/bias/temporal/{method}.tsv"
        model_scores = open(score_log_path, "r").readlines()
        model_scores = list(map(float, model_scores))
        assert len(model_scores) == len(table_scores)
        print(model_scores)
        print(table_scores)


        coef, pval = pearsonr(model_scores, table_scores)
        row = [method, coef, pval]
        table.append(row)


    print_table(table)



if __name__ == "__main__":
    main()