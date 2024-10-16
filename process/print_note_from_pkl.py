# print all the notes from a pkl file

import pickle

def print_notes_from_pkl(pkl_file):
    """Print all the notes from a .pkl file."""
    with open(pkl_file, 'rb') as filepath:
        notes = pickle.load(filepath)
    
    print(notes)

# Example of use
pkl_file = 'notes.pkl'
print_notes_from_pkl(pkl_file)
