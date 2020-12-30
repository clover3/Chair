
module = None
obj = None


def get_text_from_html(html_text: str) -> str:
    global module
    global obj
    if module is None:
        import html2text
        obj = html2text.HTML2Text()
        obj.ignore_links = True

    return obj.handle(html_text)
