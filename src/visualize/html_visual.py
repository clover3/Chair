import abc
import os
from abc import ABC

import cpath
from cpath import output_path


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


def get_tooltip_cell(s, tooltip):
    return Cell(get_tooltip_span(s, tooltip))


def set_cells_color(cells, color):
    for c in cells:
        c.target_color = color


def get_tooltip_span(span_text, tooltip_text):
    tag = "<span class=\"tooltip\">{}\
    <span class=\"tooltiptext\">{}</span>\
    </span>".format(span_text, tooltip_text)
    return tag


class VisualizerCommon(ABC):
    @abc.abstractmethod
    def write_table(self, rows):
        pass

    def multirow_print(self, cells, width=20):
        i = 0
        while i < len(cells):
            self.write_table([cells[i:i+width]])
            i += width

    def multirow_print_from_cells_list(self, cells_list, width=20):
        i = 0
        cells = cells_list[0]
        while i < len(cells):
            rows = []
            for row_idx, _ in enumerate(cells_list):
                row = cells_list[row_idx][i:i+width]
                rows.append(row)

            self.write_table(rows)
            i += width


def get_tooltip_style_text():
    return open(os.path.join(cpath.data_path, "html", "tooltip")).read()


def get_collapsible_css():
    return open(os.path.join(cpath.src_path, "html", "collapsible.css")).read()


def get_scroll_css():
    return open(os.path.join(cpath.src_path, "html", "scroll.css")).read()


def get_collapsible_script():
    return open(os.path.join(cpath.src_path, "html", "collapsible.js")).read()


class HtmlVisualizer(VisualizerCommon):
    def __init__(self, filename, dark_mode=False, use_tooltip=False, additional_styles=[]):
        p = os.path.join(output_path, "visualize", filename)
        self.f_html = open(p, "w", encoding="utf-8")
        self.dark_mode = dark_mode
        self.dark_foreground = "A9B7C6"
        self.dark_background = "2B2B2B"
        self.use_tooltip = use_tooltip
        self._write_header(additional_styles)

    def _write_header(self, additional_styles):
        self.f_html.write("<html><head>\n")
        self.f_html.write("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\"/>")
        style_dark = "body{color:#" + self.dark_foreground + ";}"
        if self.dark_mode:
            additional_styles.append(style_dark)

        if self.use_tooltip:
            tooltip_style = get_tooltip_style_text()
            additional_styles.append(tooltip_style)

        self.f_html.write("<style>")
        for style in additional_styles:
            self.f_html.write(style)
        self.f_html.write("</style>")
        self.f_html.write("</head>\n")

        if self.dark_mode:
            self.f_html.write("<body style=\"background-color:#{};\"".format(self.dark_background))
        else:
            self.f_html.write("<body>\n")

    def write_script(self, script):
        self.f_html.write("<script>")
        self.f_html.write(script)
        self.f_html.write("</script>")


    def close(self):
        self.f_html.write("</body>\n")
        self.f_html.write("</html>\n")

    def write_paragraph(self, s):
        self.f_html.write("<p>\n")
        self.f_html.write(s+"\n")
        self.f_html.write("</p>\n")

    def write_div(self, s, div_class):
        self.write_elem("div", s, div_class)

    def write_elem(self, elem, s, elem_class, style=""):
        if elem_class:
            optional_class = " class=" + elem_class
        else:
            optional_class = ""

        if style:
            style_text = " style=\"{}\"".format(style)
        else:
            style_text = ""

        self.f_html.write("<{}{}{}>\n".format(elem, optional_class, style_text))
        self.f_html.write(s + "\n")
        self.f_html.write("</{}>\n".format(elem))

    def write_div_open(self, div_class=""):
        if div_class:
            optional_class = " class=" + div_class
        else:
            optional_class = ""
        self.f_html.write("<div{}>\n".format(optional_class))

    def write_div_close(self):
        self.f_html.write("</div>\n")

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
        elif color == "Y":
            bg_color = "ffff" + ("%02x" % r)
        else:
            assert False
        return bg_color

    def get_blue_d(self, r):
        r = (0xFF - 0x2B) * r / 255
        r = 0x2B + int(r)
        bg_color = "2B2B" + ("%02x" % r)
        return bg_color


    def write_span_line(self, span_and_tooltip_list):
        if not self.use_tooltip:
            print("WARNING toolip is not activated")
        self.f_html.write("<div>")

        for span_text, tooltip_text in span_and_tooltip_list:
            self.f_html.write(get_tooltip_span(span_text, tooltip_text))
        self.f_html.write("</div>")
        self.f_html.write("<br>")


def normalize(scores):
    max_score = max(scores)
    min_score = min(scores)

    gap = max_score - min_score
    if gap < 0.001:
        gap = 1

    return [(s - min_score) / gap * 100 for s in scores]