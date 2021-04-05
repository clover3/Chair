import json
import sys

from boilerpipe.extract import Extractor


def main():
    input_path = sys.argv[1]
    save_path = sys.argv[2]

    data = json.load(open(input_path, "r"))
    out_data = {}

    for key, html in data.items():
        extractor = Extractor(extractor='ArticleExtractor', html=html)
        core_text = extractor.getText()
        out_data[key] = str(core_text)

##
    json.dump(out_data, open(save_path, "w"))


if __name__ == "__main__":
    main()