import csv
from typing import Dict, List


def write_doc(fout, doc_no, title, content):
    fout.write("<DOC>\n")
    fout.write("<DOCNO>{}</DOCNO>\n".format(doc_no))
    fout.write("<HEADLINE>{}</HEADLINE>\n".format(title))
    fout.write("<TEXT>\n")
    fout.write(content)
    fout.write("</TEXT>\n")
    fout.write("</DOC>\n")


def read_csv_as_dict(csv_path) -> List[Dict]:
    f = open(csv_path, "r")
    reader = csv.reader(f)
    data = []
    for g_idx, row in enumerate(reader):
        if g_idx == 0 :
            columns = row
        else:
            entry = {}
            for idx, column in enumerate(columns):
                entry[column] = row[idx]
            data.append(entry)

    return data

str_cord_uid = 'cord_uid'
str_title = 'title'
str_abstract = 'abstract'

def read_csv_and_write(csv_path, output_path):
    data = read_csv_as_dict(csv_path)
    fout = open(output_path, "w")
    for entry in data:
        write_doc(fout, entry[str_cord_uid], entry[str_title], entry[str_abstract])

    fout.close()



