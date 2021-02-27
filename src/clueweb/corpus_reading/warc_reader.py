import sys

# import warc
from warcio.archiveiterator import ArchiveIterator


def read_doc(record):
    payload = record.payload.read()
    idx = payload.find(b'\r\n\r\n')
    html = payload[idx:]
    html = html.strip()
    return html




def warc_gz_to_text(warc_gz_path):
    print("Processing", warc_gz_path)
    with open(warc_gz_path, 'rb') as stream:
        for record in ArchiveIterator(stream):
    # warc_file = warc.open(warc_gz_path)
    # for i, record in enumerate(warc_file):
            html = read_doc(record)
            print(html)


if __name__ == "__main__":
    warc_gz_to_text(sys.argv[1])