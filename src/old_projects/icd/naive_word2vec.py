import gensim.models
import nltk

from old_projects.icd.common import lmap, load_description


def train(save_name):
    sentences = get_sentences()
    print(len(sentences))
    model = gensim.models.Word2Vec(sentences=sentences,
                                   min_count=1)
    model.save(save_name)


def get_sentences():
    data = load_description()
    ids = lmap(lambda x:x['icd10_code'].strip(), data)
    input2 = lmap(lambda x: x['short_desc'], data)
    desc_tokens = lmap(nltk.word_tokenize, input2)

    l = []
    for id_token, desc_token in zip(ids, desc_tokens):
        l.append([id_token] + desc_token)
    return l


if __name__ == "__main__":
    save_name = "sent.w2v"
    train(save_name)