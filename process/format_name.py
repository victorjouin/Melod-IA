import os
import shutil
import re

def format_name(data_dir='data', output_dir='data_rename'):
    """Rename all files in data by keeping only the intonation and Maj or Min of the file, and add an index."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for index, file in enumerate(os.listdir(data_dir)):
        file_path = os.path.join(data_dir, file)
        intonation = re.search(r'[A-G]# ?', file)
        maj_min = re.search(r'(Maj|Min)', file, re.IGNORECASE)
        if intonation and maj_min:
            new_file = f"{index}_{intonation.group()}_{maj_min.group().capitalize()}.mid"
            new_file_path = os.path.join(output_dir, new_file)
            shutil.copy(file_path, new_file_path)
            print(f"Copied and renamed {file_path} to {new_file_path}")
        else:
            print(f"Skipped {file_path} as it does not match the required pattern")

# Example of use
data_dir = 'data'
output_dir = 'data_rename'
format_name(data_dir, output_dir)