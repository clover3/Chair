from flair.data import Sentence
from flair.models import SequenceTagger


def main():
    # load tagger
    tagger = SequenceTagger.load("flair/chunk-english")

    # make example sentence
    sentence = Sentence("The happy man has been eating at the diner")
    sentence = Sentence("We conclude that in women with preeclampsia, prolonged dietary supplementation with l-arginine significantly decreased blood pressure through increased endothelial synthesis and/or bioavailability of NO.")
    # predict NER tags
    tagger.predict(sentence)

    # print sentence
    print(sentence)

    # print predicted NER spans
    print('The following NER tags are found:')
    # iterate over entities and print
    for e in sentence._known_spans.values():
        if e.has_label('np'):
            print(e, e.annotation_layers)
        else:
            print("none")
    for entity in data_generator2.segmented_enc.sent_split_by_spacy.get_spans('np'):
        print(entity)


if __name__ == "__main__":
    main()