#remove all file from data that has a hit hat in their name

import os
import shutil
def remove_hit_hat(data_dir='data'):
    """Remove all files from data that has 'hit' or 'hat' in their name."""
    for file in os.listdir(data_dir):
        if 'hit' in file or 'hat' in file:
            file_path = os.path.join(data_dir, file)
            os.remove(file_path)
            print(f"Removed {file_path}")

# Example of use
data_dir = 'data'
remove_hit_hat(data_dir)
