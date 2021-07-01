
import csv
import os

from cpath import output_path


def main():
    save_path = os.path.join(output_path, "ca_building", "run1", "mturk_todo.csv")
    reader = csv.reader(open(save_path, "r"), delimiter=',')

    f = open(os.path.join(output_path, "ca_building", "run1", "doc_urls.html"), "w")
    f.write("<html><body><table>")
    for row in reader:
        c_text = row[0]
        p_text = row[1]
        doc_id = row[2]
        url = "https://ecc.neocities.org/clueweb/{}.html".format(doc_id)
        f.write("<tr>")
        f.write("<td><a href=\"{}\" target=\"_blank\">{}</a></td>".format(url, doc_id))
        f.write("<td>{}</td>".format(c_text))
        f.write("<td>{}</td>".format(p_text))
        f.write("</tr>")


    f.write("</table></body></html>")



if __name__ == "__main__":
    main()