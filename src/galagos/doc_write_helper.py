def write_galago_xml_doc(fout, doc_no, title, content):
    fout.write("<DOC>\n")
    fout.write("<DOCNO>{}</DOCNO>\n".format(doc_no))
    fout.write("<HEADLINE>{}</HEADLINE>\n".format(title))
    fout.write("<TEXT>\n")
    fout.write(content)
    fout.write("</TEXT>\n")
    fout.write("</DOC>\n")