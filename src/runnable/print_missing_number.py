import sys


def main():
    def find_missing_numbers(filename):
        # Read file and extract numbers
        with open(filename, 'r') as file:
            lines = file.readlines()
            numbers_in_file = {int(line.split('.')[0]) for line in lines}

        # Check for missing numbers in the range 0 to 99,999
        missing_numbers = [num for num in range(100000) if num not in numbers_in_file]

        missing_jobs = set([n//100 for n in missing_numbers])

        return missing_jobs

    # Example usage
    missing_numbers = find_missing_numbers(sys.argv[1])
    print(" ".join(map(str, missing_numbers)))
    print("Missing numbers:", missing_numbers)


if __name__ == "__main__":
    main()