import csv
import os

from cpath import output_path


def csv_subtract(file1, file2):
    reader1 = csv.reader(open(file1, "r", newline='',), delimiter=',')
    reader2 = csv.reader(open(file2, "r", newline='',), delimiter=',')

    lines_in2 = list(reader2)[1:]
    output = []
    for line in reader1:
        if line in lines_in2:
            print("skip")
        else:
            output.append(line)
    return output


def main():
    todo_all = os.path.join(output_path, "ca_building", "run1", "jobs", "mturk_todo.csv")
    shuffled_path = os.path.join(output_path, "ca_building", "run1", "jobs", "shuffled.csv")

    out_rows = csv_subtract(todo_all, shuffled_path)
    save_path = os.path.join(output_path, "ca_building", "run1", "jobs", "shuffled_remain.csv")
    writer = csv.writer(open(save_path, "w", newline=''))
    writer.writerows(out_rows)




if __name__ == "__main__":
    main()