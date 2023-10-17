
from bs4 import BeautifulSoup

def parse_table_like(file_path):
    body = open(file_path, "r").read()
    soup = BeautifulSoup(body, 'html.parser')
    texts = soup.body.findAll(text=True)
    visible_texts = filter(tag_visible, texts)


def main():
    return NotImplemented


if __name__ == "__main__":
    main()

