import re

from bert_api.client_lib import BERTClient
from cpath import at_output_dir
from list_lib import get_max_idx
from misc_lib import group_by, get_first


def load_lm_prob(keyword):
    file_pah = at_output_dir("pairing", "lm_prob_{}.txt".format(keyword))
    f = open(file_pah, "r")

    exp = r"\d*: ([^\s]*) - (.*)%"
    pattern = re.compile(exp)

    def parse_line(line):
        m = pattern.match(line)
        word = m.group(1)
        probability = m.group(2)
        e = word, float(probability)
        return e

    return list(map(parse_line, f))


def main():
    keyword = "Britain"
    lines = load_lm_prob(keyword.lower())
    prem = "Britain's best-selling tabloid, the Sun , announced as a front-page world exclusive Friday that Texan model Jerry Hall has started divorce proceedings"
    # prem = "Britain's best-selling tabloid, the Bun , announced that Texan model Jerry Hall has started divorce proceedings"
    hypo = "There is a British publication called the Sun."

    prem = prem.replace("Sun", "Dailia")
    hypo = hypo.replace("Sun", "Dailia")

    client = BERTClient("http://localhost", 8122, 300)
    threshold = 0.4
    payload = [(prem, hypo)]
    for word, prob in lines:
        new_prem = prem.replace(keyword, word)
        payload.append((new_prem, hypo))

    response = client.request_multiple(payload)
    print(payload[0], response[0])
    joined = list(zip(lines, response))
    joined.sort(key=lambda x: x[1][0])
    out_summary = []
    for (word, prob), e in joined:
        sout = e
        label_likely = list([idx for idx, score in enumerate(sout) if score > threshold])
        pred = get_max_idx(sout)
        e = word, prob, tuple(label_likely), pred
        out_summary.append(e)

    grouped = group_by(out_summary, lambda x: x[2])

    for label_likely, entries in grouped.items():
        print(label_likely)
        words = list(map(get_first, entries))
        print(" ".join(words))


def sanity():
    prem = "It is good to know."
    hypo = "I am a boy."
    client = BERTClient("http://localhost", 8122, 300)

    payload = [(prem, hypo)]
    response = client.request_multiple(payload)
    for (p, h), e in zip(payload, response):
        sout = e
        print(p, h, sout)


if __name__ == "__main__":
    main()