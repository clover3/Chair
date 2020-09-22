from visualize.html_visual import HtmlVisualizer


def main():
    html = HtmlVisualizer("tooltip_test.html", dark_mode=False, use_tooltip=True)

    line = [
        ("1", "hello"),
        ("2", "word")
    ]
    html.write_span_line(line)
    html.write_span_line(line)


if __name__ == "__main__":
    main()