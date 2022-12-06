from explain.bert_components.cls_probe_base_visualize import write_html
from explain.bert_components.cmd_nli import ModelConfig
from explain.bert_components.load_nli_dev import load_data
from explain.bert_components.load_probe import load_probe
from visualize.html_visual import HtmlVisualizer


def main():
    model_config = ModelConfig()
    model, bert_cls_probe = load_probe(model_config)
    batches = load_data(model_config)
    x0, x1, x2, y = batches[0]
    X = (x0, x1, x2)
    save_name = "cls_probe_base_local.html"
    html = HtmlVisualizer(save_name)
    logits, probes = bert_cls_probe(X)
    write_html(html, x0, logits, probes, y)


if __name__ == "__main__":
    main()


