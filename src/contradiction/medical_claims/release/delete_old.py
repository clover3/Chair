import os



def remove_old_files_recursively(directory_path):
    """
    Recursively remove files from the given directory and its subdirectories that end with 'old'.

    Parameters:
    directory_path (str): Path to the directory where files need to be checked and removed.
    """
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith("old"):
                # Construct full path
                file_path = os.path.join(root, file)
                # Remove the file
                os.remove(file_path)
                print(f"Removed file: {file_path}")

# Example usage
# directory_path = "path/to/your/directory"
# remove_old_files(directory_path)


def main():
    # Example usage
    target_root = "C:\work\code\cond-nli_work\src"
    remove_old_files_recursively(target_root)


if __name__ == "__main__":
    main()
