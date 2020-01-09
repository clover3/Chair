import os

from cpath import output_path
from crs.load_stance_annotation import load_stance_annot
from elastic.retrieve import get_comment, get_paragraph, get_title


def visulize(path):
    r = load_stance_annot(path)
    #r.sort(key=lambda x:x['statement'])

    f = open(os.path.join(output_path, "crs", "visulize1.html"), "w")
    f.write('<html><head>'
            '<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">'
            '</head><body>')

    last_statement = ""
    def code2str(code):
        return {0:'Not Found', 1:'Direct', 2:'Implies'}[code]

    def print_evidence(doc_id, seg_id):
        f.write("<br>")
        title = get_title(doc_id)
        title_html = "<div><i>" + title + "</i></div>"
        badge =""
        seg_id_i = int(seg_id)
        if seg_id_i < 1000:
            badge = "<span class=\"badge badge-secondary\">article</span><br>"
            p = get_paragraph(doc_id, seg_id_i)
        else:
            badge = "<span class=\"badge badge-primary\">comment</span>"
            p = get_comment(doc_id, seg_id_i)

        f.write(title_html)
        f.write(badge)
        f.write(p)
        f.write("<br>")
    f.write("<div class=\"container\">")
    for e in r:

        statement =  e['statement']

        f.write("<div class=\"bg-light my-2\">")
        if statement != last_statement:
            f.write("<div class=\"alert alert-primary\"> statement : " + statement + "</div>")
        f.write("<div class=\"row\">")
        f.write("<div class=\"col\">")
        f.write("<div class=\"btn btn-success\"> Support : " + code2str(e['support']) + "</div>")
        f.write("<br>")

        if len(e['support_evidence']) > 1:
            doc_id, seg_id = e['support_evidence']
            f.write("Support Evidence: " + doc_id + ", " + seg_id)
            print_evidence(doc_id, seg_id)
        f.write("</div>")

        f.write("<div class=\"col\">")
        f.write("<div class=\"btn btn-danger\"> Dispute : " + code2str(e['dispute']) + "</div>")
        f.write("<br>")

        if len(e['dispute_evidence']) > 1:
            doc_id, seg_id = e['dispute_evidence']
            f.write("Dispute Evidence: " + doc_id + ", " + seg_id)
            print_evidence(doc_id, seg_id)

        f.write("</div>")

        f.write("</div>")
        f.write("</div>")
        last_statement = statement
    f.write("</div>")
    f.write("</body></html>")


def visulize4():
    path = "C:\work\Data\CKB annotation\Search Stances 4\\Batch_3749275_batch_results.csv"
    visulize(path)

def visulize1():
    path = "C:\work\Data\CKB annotation\dipsute annotation 1\\Batch_3746208_batch_results.csv"
    visulize(path)


visulize4()
