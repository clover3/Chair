import re
import sys
from cpath import output_path
from misc_lib import path_join


def check_file_ending(filename):
    with open(filename, 'r') as f:
        # Read all lines
        lines = f.readlines()

        # Ensure there are at least 2 lines
        if len(lines) < 2:
            return None, None

        # Get the last 2 lines
        second_last_line = lines[-2].strip()
        last_line = lines[-1].strip()

        # Check if the lines match the required pattern
        match = re.match(r'Now reporting task :  run_candidate_(\d+)_(\d+)$', second_last_line)
        if match and last_line == "Done":
            # Extract numbers from the matched groups
            num1, num2 = map(int, match.groups())
            return num1, num2

    return None, None


def main():

    start = 644803
    end = 645394
    print(start)
    start_set = set()
    for job_no in range(start, end):
        file_path = path_join(output_path, "log", "{}.txt".format(job_no))
        try:
            ret = check_file_ending(file_path)
            num1, num2 = ret
            if num1 is not None and num2 is not None:
                start_set.add(num1)
                # print(f"{job_no} The file ends with the required lines having numbers: {num1} and {num2}.")
                print(job_no, num1)
            else:
                pass
                # print(f"{job_no} The file does NOT end with the required lines.")
        except FileNotFoundError:
            pass



if __name__ == "__main__":
    main()