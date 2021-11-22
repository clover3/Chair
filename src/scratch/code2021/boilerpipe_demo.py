from boilerpipe.extract import Extractor


def main2():
    html_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\html\\clueweb12-0000tw-15-01092.html"
    html = open(html_path, "r", encoding="utf-8").read()
    extractor_list = [
        "DefaultExtractor",
        "ArticleExtractor",
        "ArticleSentencesExtractor",
        "KeepEverythingExtractor",
        "KeepEverythingWithMinKWordsExtractor",
        "LargestContentExtractor",
        "NumWordsRulesExtractor",
        "CanolaExtractor",
    ]

    for option in extractor_list:
        extractor = Extractor(extractor=option, html=html)
        core_text = extractor.getText()

        print(">> {}".format(option))
        print(core_text.split("\n"))


def main():
    html_path = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\html\\clueweb12-0000tw-15-01092.html"
    html = open(html_path, "r", encoding="utf-8").read()
    extractor = Extractor(extractor="ArticleSentencesExtractor", html=html)


if __name__ == "__main__":
    main2()