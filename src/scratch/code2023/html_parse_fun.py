import os

from bs4 import BeautifulSoup


def parse_student_html_first_col(file_path):
    html = open(file_path, "r").read()
    soup = BeautifulSoup(html, 'html.parser')

    # Find the <a> tag within the <th> tag and extract the text
    rows = soup.find_all('tr')
    names = []
    for row in rows:
        try:
            # Extract the name from each row
            name_tag = row.find('th', align="Left").a
            if name_tag:  # Check if the anchor tag exists in the 'th' tag
                names.append(name_tag.text)
        except AttributeError:
            pass

    return set(names)


def parse_grad_student_file(file_path):
    html = open(file_path, "r").read()
    soup = BeautifulSoup(html, 'html.parser')
    # Find the <a> tag within the <th> tag and extract the text
    rows = soup.find_all('tr')
    names_w_phone = []
    names_wo_phone = []
    for row in rows:
        try:
            # Extract the name from each row
            name_tag = row.find('th', align="Left")
            if name_tag:  # Check if the anchor tag exists in the 'th' tag
                col1, col2, col3 = row.find_all('th', align="Left")
                name = col1.text
                phone = col2.text
                if phone.strip():
                    names_w_phone.append(name)
                else:
                    names_wo_phone.append(name)

        except AttributeError:
            pass

    return set(names_w_phone), set(names_wo_phone)


def parse_grad_student_file2017(file_path):
    html = open(file_path, "r").read()
    soup = BeautifulSoup(html, 'html.parser')
    # Find the <a> tag within the <th> tag and extract the text
    rows = soup.find_all('tr')
    phd = []
    non_phd = []
    for row in rows:
        try:
            # Extract the name from each row
            name_tag = row.find('th', align="Left")

            if name_tag:  # Check if the anchor tag exists in the 'th' tag
                col1, col2, col3, col4 = row.find_all('th', align="Left")
                name = col1.text
                program = col2.text
                if "phd" in program.lower():
                    phd.append(name)
                else:
                    non_phd.append(name)

        except AttributeError:
            pass

    return set(phd), set(non_phd)


def get_2016_new_phd():
    dir_path = "C:\work\code\cics_student"
    old_grad_path_d = {
        "2015": "2015_09_grad.html",
        "2016": "2016_09_grad.html",
    }
    def parse_pre2017_file(file_name):
        return parse_student_html_first_col(os.path.join(dir_path, file_name))

    output = {k: parse_pre2017_file(v) for k, v in old_grad_path_d.items()}
    phd2017, non_phd2017 = parse_grad_student_file2017(os.path.join(dir_path, "grad_student_2017_Aug.html"))

    new_grad2016 = output["2016"] - output["2015"]
    #
    # print("new_grad2016", len(new_grad2016))
    # print("phd2017", len(phd2017))
    # print("non_phd2017", len(non_phd2017))

    phd2016 = [t for t in new_grad2016 if t in phd2017]
    non_phd2016 = [t for t in new_grad2016 if t in non_phd2017]
    unknown2016 = [t for t in new_grad2016 if t not in non_phd2017 and t not in phd2017]
    # print("phd2016", len(phd2016))
    # print("non_phd2016", len(non_phd2016))
    # print("unknown2016", len(unknown2016))
    return set(phd2016)


def parse_print():
    dir_path = "C:\work\code\cics_student"
    path_d = {
        "2016": "graduate_student_2016.html",
        "2017": "2017_10.html",
        "2018": "2018_10.html",
        "2019": "2020_03.html",
        "2020": "2020_09.html",
        "2021": "2021_10.html",
        "2022": "2022_10.html",
        "2023": "2023_10.html",
    }

    def parse_file(file_name):
        return parse_student_html_first_col(os.path.join(dir_path, file_name))

    output = {k: parse_file(v) for k, v in path_d.items()}

    print("Existing student")
    for k, v in output.items():
        print(k, len(v))

    new_2017 = output["2017"] - output["2016"]
    print(f"{len(new_2017)} new phd 2017")

    print("time / remain / disappear")
    for time_i in range(2018, 2024):
        point = str(time_i)
        disappear = new_2017 - output[point]
        remain = new_2017.intersection(output[point])
        print(point, len(remain), len(disappear))

    new_phd2016 = get_2016_new_phd()
    print(f"{len(new_phd2016)} new phd 2016")
    print("time / remain / disappear")
    for time_i in range(2018, 2024):
        point = str(time_i)
        disappear = new_phd2016 - output[point]
        remain = new_phd2016.intersection(output[point])
        print(point, len(remain), len(disappear))


if __name__ == "__main__":
    parse_print()

