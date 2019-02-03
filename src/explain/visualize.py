


def print_color_html(word, r):
    r = 255 -r
    bg_color = ("%02x" % r) + ("%02x" % r) + "ff"

    html = "<td bgcolor=\"#{}\">&nbsp;{}&nbsp;</td>".format(bg_color, word)
    #    html = "<td>&nbsp;{}&nbsp;</td>".format(word)
    return html

def visualize(result, out_name):
    f = open("../{}.html".format(out_name), "w")

    f.write("<html>")
    f.write("<body>")
    f.write("<div width=\"400\">")
    for entry in result:
        pred_p, pred_h, prem, hypo = entry

        max_score = max(max(pred_p), max(pred_h))
        min_score = min(min(pred_p), min(pred_h))
        print(max_score)
        cut = max_score * 0.5
        print(cut)

        f.write("<tr>")
        for display_name, tokens, scores in [("Premise", prem, pred_p), ("Hypothesis", hypo, pred_h)]:
            f.write("<td><b>{}<b></td>\n".format(display_name))
            f.write("<table style=\"border:1px solid\">")
            for i, token in enumerate(tokens):
                print("{}({}) ".format(token, scores[i]), end="")

                r = int((scores[i] - min_score) * 255 / (max_score - min_score))

                f.write(print_color_html(token, r))
            print()
            f.write("</tr>")
            f.write("</tr></table>")

        f.write("</tr>")

        f.write("</div><hr>")

    f.write("</div>")
    f.write("</body>")
    f.write("</html>")