from data_generator.common import get_tokenizer
from misc_lib import get_dir_files
from tf_util.enum_features import load_record
from visualize.html_visual import HtmlVisualizer
from visualize.show_feature import write_feature_to_html


def main():
    html = HtmlVisualizer("tf_rel_filter.html")
    tokenizer = get_tokenizer()

    path = "/mnt/nfs/work3/youngwookim/data/bert_tf/tf_rel_filter_B_dev/"

    def itr():
        for file in get_dir_files(path):
            for item in load_record(file):
                yield item

    for feature in itr():
        write_feature_to_html(feature, html, tokenizer)

main()