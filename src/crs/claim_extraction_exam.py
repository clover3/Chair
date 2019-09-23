from elastic.insert_comment import load_guardian_uk_comments
from crawl.guardian_uk import load_commented_articles_opinion
from summarization.text_rank import TextRank
from summarization.tokenizer import tokenize
from misc_lib import flatten
from cie.claim_gen import generate


def syntactic_parsing_method(article, comments):
    all_texts = article + comments
    all_tokens = list([tokenize(t, set()) for t in all_texts])
    tr = TextRank(all_tokens)
    r = tr.run(flatten(all_tokens))
    r = generate(all_texts, r)
    print(r)


def load_data():
    articles = load_commented_articles_opinion()
    comments = load_guardian_uk_comments()

    comments_d = {}
    for c in comments:
        short_id = c['short_id']
        comments_d[short_id] = c

    data = []
    for article in articles:
        id, title, short_id, infos = article
        paras = infos['paragraphs']


        comment_texts = []
        for t in comments_d[short_id]['comments']:
            head, tail = t
            comment_texts.append(head[1])
            for c in tail:
                comment_texts.append(c[1])

        entry = short_id, paras, comment_texts
        data.append(entry)
    return data




def run_experiment():
    data = load_data()
    short_id, paras, comment_texts = data[0]
    syntactic_parsing_method(paras, comment_texts)




if __name__ == "__main__":
    run_experiment()