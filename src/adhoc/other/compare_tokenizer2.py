import nltk
from data_generator.tokenizer_wo_tf import get_tokenizer



def main():
    tokenizer = get_tokenizer()
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    analyzer = Analyzer(get_lucene_analyzer(stemmer='krovetz'))
    tokenizer_d = {
        "NLTK": nltk.tokenize.word_tokenize,
        "Bert": tokenizer.basic_tokenizer.tokenize,
        "lucene": analyzer.analyze,
    }
    t = "Good muffins cost $300,000.88\nin New York. Please buy me\ntwo of them.\nThanks"
    for name, fn in tokenizer_d.items():
        print(name, "\t", fn(t))


if __name__ == "__main__":
    main()