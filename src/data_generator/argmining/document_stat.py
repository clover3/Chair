from misc_lib import average
from data_generator.argmining.ukp import all_topics
from data_generator.data_parser.ukp import load
from collections import defaultdict, Counter

import os
import path


def stat():
    all_data = {}
    for topic in all_topics:
        data = load(topic)
        key = 'retrievedUrl'
        d = defaultdict(list)
        for entry in data:
            d[entry[key]].append(entry)

        print(topic)
        print("# sentence : ", len(data))
        print("# urls : ", len(list(d.keys())))

        length = list([len(v) for k, v in d.items()])

        print("max min avg")
        print(max(length), min(length), average(length))
        all_data[topic] = d

    return all_data


def load_with_doc():
    all_data = {}
    for topic in all_topics:
        data = load(topic)
        key = 'retrievedUrl'
        d = defaultdict(list)
        for entry in data:
            d[entry[key]].append(entry)

        all_data[topic] = d
    return all_data


def visualize():
    labels = ["NoArgument", "Argument_for", "Argument_against"]
    all_data = load_with_doc()
    for topic in all_topics:
        print(topic)
        target_topic = topic
        topic_data = all_data[topic]

        f_html = open(os.path.join(path.output_path, "visualize", "stance_{}_gold_doc.html".format(topic)), "w")
        f_html.write("<html><head>\n")

        #tooptip_style = open(os.path.join(path.data_path, "html", "tooltip")).read()
        #f_html.write(tooptip_style)
        f_html.write("</head>\n")
        f_html.write("<h4>{}<h4>\n".format(target_topic))

        for key in topic_data:
            url = key

            sents =[]
            topic_stances = []
            for entry in topic_data[key]:
                y = labels.index(entry['annotation'])
                sents.append(entry['sentence'])
                topic_stances.append(y)
            print("")

            num_total = len(topic_data[key])
            count = Counter(topic_stances)

            p1 = count[1] / len(topic_stances)
            p2 = count[2] / len(topic_stances)
            f_html.write("<br>")
            f_html.write("<div>")
            f_html.write(url)
            f_html.write("</div>")
            f_html.write("<div>")
            f_html.write("<span>{0:.2f},{1:.2f}&nbsp;&nbsp;</span>".format(p1, p2))

            for i, stance in enumerate(topic_stances):
                tag = "<span class=\"tooltip\">{}\
                <span class=\"tooltiptext\">{}</span>\
                </span>".format(stance, sents[i])

                tag = "<div><span>{}</span>&nbsp;<span>{}</span></div>".format(stance, sents[i])

                f_html.write(tag + "\n")
            f_html.write("</div>")
            f_html.write("<br>")
            print(topic_stances)
        f_html.write("\n</html>")


def print_url_list():
    all_data = load_with_doc()
    for topic in all_topics:
        print(topic)
        topic_data = all_data[topic]
        for url in topic_data:
            print(url)



if __name__ == "__main__":
    visualize()
