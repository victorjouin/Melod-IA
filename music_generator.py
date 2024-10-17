import os
import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
import tensorflow as tf

def get_notes(data_dir='data'):
    notes = []
    bad_files = []

    for file in os.listdir(data_dir):
        if file.lower().endswith(('.mid', '.midi')):
            file_path = os.path.join(data_dir, file)
            try:
                midi = converter.parse(file_path)
                print(f"Analyse de {file}")

                notes_to_parse = None

                try:
                    s2 = instrument.partitionByInstrument(midi)
                    notes_to_parse = s2.parts[0].recurse()
                except:
                    notes_to_parse = midi.flat.notes

                for element in notes_to_parse:
                    if isinstance(element, note.Note):
                        notes.append(str(element.pitch))
                    elif isinstance(element, chord.Chord):
                        chord_str = '.'.join(str(p) for p in element.pitches)
                        notes.append(chord_str)
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {file}: {e}")
                bad_files.append(file_path)

    with open('notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, sequence_length=10):
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        input_sequence = [note_to_int[note] for note in sequence_in]
        output_note = note_to_int[sequence_out]
        network_input.append(input_sequence)
        network_output.append(output_note)

    n_patterns = len(network_input)

    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitchnames))
    network_output = tf.keras.utils.to_categorical(network_output, num_classes=len(pitchnames))

    return network_input, network_output, pitchnames, note_to_int, int_to_note

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def train_network():
    notes = get_notes()
    network_input, network_output, pitchnames, note_to_int, int_to_note = prepare_sequences(notes)
    n_vocab = len(pitchnames)
    model = create_network(network_input, n_vocab)
    model.fit(network_input, network_output, epochs=100, batch_size=64)
    model.save('model.h5')

def generate_music(generate_length=500, temperature=1.0, output_file='output.mid'):
    with open('notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)
    network_input, _, pitchnames, note_to_int, int_to_note = prepare_sequences(notes)
    n_vocab = len(pitchnames)
    model = load_model('model.h5')
    prediction_output = generate_notes(model, network_input, int_to_note, n_vocab, temperature, generate_length)
    create_midi(prediction_output, output_file=output_file)

def generate_notes(model, network_input, int_to_note, n_vocab, temperature=1.0, generate_length=500):
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(generate_length):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], 1))
        prediction = model.predict(prediction_input, verbose=0)

        # Application de la température
        prediction = np.log(prediction + 1e-9) / temperature
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)

        # Sélection de l'index
        index = np.random.choice(range(n_vocab), p=prediction.ravel())
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index / float(n_vocab))
        pattern = pattern[1:]

    return prediction_output

def create_midi(prediction_output, output_file='output.mid'):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern):
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5  # Vous pouvez ajuster cette valeur

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Modèle LSTM pour la génération de mélodies MIDI avec accords.')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle LSTM.')
    parser.add_argument('--generate', action='store_true', help='Générer une nouvelle mélodie.')
    parser.add_argument('--output_file', type=str, default='output.mid', help='Nom du fichier MIDI de sortie.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Température pour la génération de notes.')
    parser.add_argument('--length', type=int, default=500, help='Longueur de la séquence générée.')

    args = parser.parse_args()

    if args.train:
        train_network()
    if args.generate:
        generate_music(generate_length=args.length, temperature=args.temperature, output_file=args.output_file)
