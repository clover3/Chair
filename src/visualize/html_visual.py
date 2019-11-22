import os
from path import output_path

def normalize255(v, max):
    if max==0:
        return 0
    return v/max * 255


def get_color(r):
    r = 255 - int(r)
    bg_color = ("%02x" % r) + ("%02x" % r) + "ff"
    return bg_color


class Cell:
    def __init__(self, s, highlight_score=0, space_left=True, space_right=True):
        self.s = str(s)

        # score should be normalized to 0~255 scale, or else floor
        if highlight_score > 255:
            highlight_score = 255
        elif highlight_score < 0:
            highlight_score = 0
        self.highlight_score = highlight_score
        self.space_left = space_left
        self.space_right = space_right


class HtmlVisualizer:
    def __init__(self, filename, dark_mode=False):
        p = os.path.join(output_path, "visualize", filename)
        self.f_html = open(p, "w", encoding="utf-8")
        self.dark_mode = dark_mode
        self.dark_foreground = "A9B7C6"
        self.dark_background = "2B2B2B"
        self._write_header()

    def _write_header(self):
        self.f_html.write("<html><head>\n")

        if self.dark_mode:
            self.f_html.write("<style>body{color:#" + self.dark_foreground + ";}</style>")

        self.f_html.write("</head>\n")

        if self.dark_mode:
            self.f_html.write("<body style=\"background-color:#{};\"".format(self.dark_background))
        else:
            self.f_html.write("<body>\n")

    def close(self):
        self.f_html.write("</body>\n")
        self.f_html.write("</html>\n")

    def write_paragraph(self, s):
        self.f_html.write("<p>\n")
        self.f_html.write(s+"\n")
        self.f_html.write("</p>\n")

    def write_headline(self, s, level=4):
        self.f_html.write("<h{}>{}</h{}>\n".format(level, s, level))

    def write_table(self, rows):
        self.f_html.write("<table>\n")

        for row in rows:
            self.f_html.write("<tr>\n")
            for cell in row:
                s = self.get_cell_html(cell)
                self.f_html.write(s)
            self.f_html.write("</tr>\n")
        self.f_html.write("</table>\n")

    def get_cell_html(self, cell):
        left = "&nbsp;" if cell.space_left else ""
        right = "&nbsp;" if cell.space_right else ""
        if cell.highlight_score:
            if not self.dark_mode:
                bg_color = self.get_blue(cell.highlight_score)
            else:
                bg_color = self.get_blue_d(cell.highlight_score)

            s = "<td bgcolor=\"#{}\">{}{}{}</td>".format(bg_color, left, cell.s, right)
        else:
            s = "<td>{}{}{}</td>".format(left, cell.s, right)

        return s

    def get_blue(self, r):
        r = 255 - int(r)
        bg_color = ("%02x" % r) + ("%02x" % r) + "ff"
        return bg_color

    def get_blue_d(self, r):
        r = (0xFF - 0x2B) * r / 255
        r = 0x2B + int(r)
        bg_color = "2B2B" + ("%02x" % r)
        return bg_color