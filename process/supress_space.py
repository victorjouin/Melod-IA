# remove space in data directory file's name

import os
import shutil

def remove_space(data_dir='data'):
    """Remove space in data directory file's name"""
    for file in os.listdir(data_dir):
        if ' ' in file:
            old_file = os.path.join(data_dir, file)
            new_file = os.path.join(data_dir, file.replace(' ', '_'))
            os.rename(old_file, new_file)
            print(f"Renamed: {old_file} to {new_file}")

if __name__ == '__main__':
    remove_space()
        