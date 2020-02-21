import os

from cpath import output_path


class TexTableVisualizer:
    def __init__(self, filename, dark_mode=False):
        p = os.path.join(output_path, "visualize", filename)
        self.f_writer = open(p, "w", encoding="utf-8")
        self.dark_mode = dark_mode
        self._write_header()

    def _write_header(self):
        pass

    def begin_table(self):
        texts = "\\begin{table} \\begin{flushleft}\n"\
                + "\setlength{\\tabcolsep}{0.5em} % for the horizontal padding"
        self.f_writer.write(texts)

    def close_table(self):
        self.f_writer.write("\end{flushleft}\end{table}\n\n\n")

    def write_paragraph(self, s):
        self.f_writer.write(s + "\n")

    def write_headline(self, s, level=4):
        self.f_writer.write("\section{%s}\n" % s)

    def write_table(self, rows):
        width = max([len(row) for row in rows])
        line_scale_begin = "\scalebox{0.7}{\n"
        line_scale_end = "}\n"
        self.f_writer.write(line_scale_begin)

        line = "\\begin{tabular}{" + "c" * width + "}\n"
        self.f_writer.write(line)

        for row in rows:
            s = "&".join([self.get_cell_tex(cell) for cell in row])
            self.f_writer.write(s)
            self.f_writer.write("\\\\ \n")
        self.f_writer.write("\\end{tabular}\n")
        self.f_writer.write(line_scale_end)

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

    def get_cell_tex(self, cell):
        left = "" if cell.space_left else ""
        right = "" if cell.space_right else ""
        if cell.highlight_score:
            color_str = self.get_color_str(cell.target_color)

            highlight_score = cell.highlight_score*0.5 if cell.target_color != "Y" else cell.highlight_score
            color_val = int(highlight_score)
            s = "\cellcolor{%s!%d}%s%s%s" % (color_str, color_val, left, cell.s, right)
        else:
            s = "{}{}{}".format(left, cell.s, right)

        return s

    def get_color_str(self, color):
        if color == "B":
            return "blue"
        elif color == "R":
            return "red"
        elif color == "G":
            return "green"
        elif color == "Y":
            return "yellow"
        else:
            assert False


class TexTableNLIVisualizer:
    def __init__(self, filename, dark_mode=False):
        p = os.path.join(output_path, "visualize", filename)
        self.f_writer = open(p, "w", encoding="utf-8")
        self.dark_mode = dark_mode
        self._write_header()

    def _write_header(self):
        pass

    def write_instance(self, pred_str, p_rows, h_rows):
        texts = "\\begin{tabular}{cl}\n"
        texts += "\\multirow{2}{*}{%s} & " % pred_str

        for idx, row in enumerate(p_rows):
            if idx != 0 :
                texts += "&"
            texts += self.get_table_code(row)

        self.f_writer.write(texts)

    def begin_table(self):
        texts = "\\begin{table} \\begin{flushleft}\n"\
                + "\setlength{\\tabcolsep}{0.5em} % for the horizontal padding"
        self.f_writer.write(texts)

    def close_table(self):
        self.f_writer.write("\end{flushleft}\end{table}\n\n\n")

    def write_paragraph(self, s):
        self.f_writer.write(s + "\n")

    def write_headline(self, s, level=4):
        self.f_writer.write("\section{%s}\n" % s)

    def get_table_row_code(self, rows):
        width = max([len(row) for row in rows])
        texts = ""
        line_scale_begin = "\scalebox{0.7}{\n"
        line_scale_end = "}\n"
        texts += line_scale_begin

        texts += "\\begin{tabular}{" + "c" * width + "}\n"

        for row in rows:
            s = "&".join([self.get_cell_tex(cell) for cell in row])
            texts += s
            texts += "\\\\ \n"
        texts += "\\end{tabular}\n"
        texts += line_scale_end
        return texts

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

    def get_cell_tex(self, cell):
        left = "" if cell.space_left else ""
        right = "" if cell.space_right else ""
        if cell.highlight_score:
            color_str = self.get_color_str(cell.target_color)

            highlight_score = cell.highlight_score*0.5 if cell.target_color != "Y" else cell.highlight_score
            color_val = int(highlight_score)
            s = "\cellcolor{%s!%d}%s%s%s" % (color_str, color_val, left, cell.s, right)
        else:
            s = "{}{}{}".format(left, cell.s, right)

        return s

    def get_color_str(self, color):
        if color == "B":
            return "blue"
        elif color == "R":
            return "red"
        elif color == "G":
            return "green"
        elif color == "Y":
            return "yellow"
        else:
            assert False



class TexNLIVisualizer:
    def __init__(self, filename, dark_mode=False):
        p = os.path.join(output_path, "visualize", filename)
        self.f_writer = open(p, "w", encoding="utf-8")
        self.dark_mode = dark_mode
        self._write_header()

    def _write_header(self):
        pass

    def close(self):
        pass

    def write_paragraph(self, s):
        self.f_writer.write(s + "\n")

    def write_headline(self, s, level=4):
        self.f_writer.write("\section{%s}\n" % s)

    def begin_table(self):
        texts = "\\begin{table} \\begin{flushleft}\n"
        texts += "\\begin{tabular}{p{0.13\\textwidth}p{0.85\\textwidth}}\n"
        texts += "\\toprule \n"
        texts += "\multicolumn{1}{c}{Prediction} & \multicolumn{1}{c}{Sentences} \\\\ \hline\n"
        self.f_writer.write(texts)

    def close_table(self):
        self.f_writer.write("\\bottomrule")
        self.f_writer.write("\\end{tabular}\n")
        self.f_writer.write("\end{flushleft}\end{table}\n\n\n")

    def write_instance(self, pred_str, rows):

        left_column = "\multirow{2}{*}{\\begin{tabular}[c]{@{}c@{}}%s \end{tabular}}" % (pred_str)
        left_column = "\multicolumn{1}{c}{" + left_column + "}"
        self.f_writer.write(left_column+"\n")

        for idx, row in enumerate(rows):
            horizontal_loc = 0
            s = " & "
            for cell in row:
                s += self.get_cell_tex(cell)
                move = len(cell.s) + (1 if cell.space_left else 0) + (1 if cell.space_right else 0)
                horizontal_loc += move

                print(cell.s, horizontal_loc)
                if horizontal_loc > 90:
                    s += "\n"
                    horizontal_loc = 0

            self.f_writer.write(s)
            self.f_writer.write("\\\\ \n")
        self.f_writer.write("\hline\n")

    def get_cell_tex(self, cell):
        left = " " if cell.space_left else ""
        right = " " if cell.space_right else ""
        text = cell.s
        text = text.replace("$", "\\$")
        if cell.highlight_score:
            color_str = self.get_color_str(cell.target_color)
            highlight_score = cell.highlight_score*0.5 if cell.target_color != "Y" else cell.highlight_score
            color_val = int(highlight_score)
            s = "{\setlength{\\fboxsep}{0pt}\colorbox{%s!%d}{\strut{}%s%s%s}}" % (color_str, color_val, left, text, right)
        else:
            s = "{}{}{}".format(left, text, right)

        return s

    def get_color_str(self, color):
        if color == "B":
            return "blue"
        elif color == "R":
            return "red"
        elif color == "G":
            return "green"
        elif color == "Y":
            return "yellow"
        else:
            assert False
