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
    def __init__(self, s, highlight_score=0, space_left=True, space_right=True, target_color="B"):
        if type(s) == float:
            self.s = "{:02.2f}".format(s)
        else:
            self.s = str(s)

        # score should be normalized to 0~255 scale, or else floor
        if highlight_score > 255:
            highlight_score = 255
        elif highlight_score < 0:
            highlight_score = 0
        self.highlight_score = highlight_score
        self.space_left = space_left
        self.space_right = space_right
        self.target_color = target_color


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
        self.f_html.write("<table style=\"border-spacing: 0px;\">\n")

        for row in rows:
            self.f_html.write("<tr>\n")
            for cell in row:
                s = self.get_cell_html(cell)
                self.f_html.write(s)
            self.f_html.write("</tr>\n")
        self.f_html.write("</table>\n")

    def multirow_print(self, cells, width=20):
        i = 0
        while i < len(cells):
            self.write_table([cells[i:i+width]])
            i += width

    def get_cell_html(self, cell):
        left = "&nbsp;" if cell.space_left else ""
        right = "&nbsp;" if cell.space_right else ""
        no_padding = "style=\"padding-right:0px; padding-left:0px\""
        if cell.highlight_score:
            if not self.dark_mode:
                bg_color = self.get_color(cell.highlight_score, cell.target_color)
            else:
                bg_color = self.get_blue_d(cell.highlight_score)

            s = "<td bgcolor=\"#{}\" {}>{}{}{}</td>".format(bg_color, no_padding, left, cell.s, right)
        else:
            s = "<td {}>{}{}{}</td>".format(no_padding, left, cell.s, right)

        return s

    def get_color(self, score, color):
        r = 255 - int(score)
        if color == "B":
            bg_color = ("%02x" % r) + ("%02x" % r) + "ff"
        elif color == "R":
            bg_color = "ff" + ("%02x" % r) + ("%02x" % r)
        elif color == "G":
            bg_color = ("%02x" % r) + "ff" + ("%02x" % r)
        else:
            assert False
        return bg_color

    def get_blue_d(self, r):
        r = (0xFF - 0x2B) * r / 255
        r = 0x2B + int(r)
        bg_color = "2B2B" + ("%02x" % r)
        return bg_color