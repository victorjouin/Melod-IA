# Script complet pour l'entraînement et l'utilisation d'un modèle LSTM pour la génération de mélodies
# Auteur : [Votre nom]
# Date : [Date]

import os
import shutil
import numpy as np
import pickle

from music21 import converter, instrument, note, chord, stream


# Pour le modèle LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation
from tensorflow.keras.utils import to_categorical

import os
import tensorflow as tf

# Activer les logs détaillés de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
tf.debugging.set_log_device_placement(True)

# Configurer TensorFlow pour allouer la mémoire GPU de manière dynamique

gpus = tf.config.experimental.list_physical_devices('GPU')
print("GPUs disponibles:", gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "GPU Physique,", len(logical_gpus), "GPU Logique")
    except RuntimeError as e:
        print(e)


# Fonction pour collecter les fichiers MIDI
def collect_mid_files(source_dir, target_dir='data'):
    """
    Recherche récursivement tous les fichiers .mid dans source_dir et ses sous-répertoires,
    et les copie dans target_dir.
    """
    # Vérifier si le répertoire cible existe, sinon le créer
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Parcourir récursivement le répertoire source
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
                source_file = os.path.join(root, file)
                destination_file = os.path.join(target_dir, file)
                
                # Gérer les conflits de nom de fichier en ajoutant un suffixe si nécessaire
                count = 1
                base_name, extension = os.path.splitext(file)
                while os.path.exists(destination_file):
                    destination_file = os.path.join(target_dir, f"{base_name}_{count}{extension}")
                    count += 1
                
                shutil.copy2(source_file, destination_file)
                print(f'Copié: {source_file} vers {destination_file}')

# Fonction pour extraire les notes des fichiers MIDI
def get_notes(data_dir='data'):
    """Extrait les notes des fichiers MIDI dans le répertoire data."""
    notes = []
    bad_files = []

    for file in os.listdir(data_dir):
        if file.lower().endswith('.mid') or file.lower().endswith('.midi'):
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
                        # Représenter un accord par une notation unique
                        notes.append('.'.join(str(n) for n in element.normalOrder))
            except Exception as e:
                print(f"Erreur lors du traitement du fichier {file}: {e}")
                bad_files.append(file_path)

    # Sauvegarde des notes pour une utilisation future
    with open('notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)

    if bad_files:
        print("\nLes fichiers suivants n'ont pas pu être traités:")
        for bf in bad_files:
            print(bf)

    return notes

# Fonction pour préparer les séquences pour le modèle
def prepare_sequences(notes, n_vocab, sequence_length=100):
    """Prépare les entrées et sorties pour le modèle LSTM."""
    pitchnames = sorted(set(notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # Création des séquences d'entrée et de sortie
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape pour LSTM
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalisation
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return (network_input, network_output)

# Fonction pour créer le modèle LSTM
def create_network(network_input, n_vocab):
    """Crée le modèle LSTM pour la génération de musique."""
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# Fonction pour entraîner le modèle
def train(model, network_input, network_output, epochs=100, batch_size=64):
    """Entraîne le modèle."""
    model.fit(network_input, network_output, epochs=epochs, batch_size=batch_size)
    # Sauvegarder le modèle
    model.save('model.h5')

# Fonction principale pour l'entraînement
def train_network():
    """Prépare les données et entraîne le modèle."""
    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    train(model, network_input, network_output)

# Fonction pour générer des notes
def generate_notes(model, network_input, pitchnames, n_vocab, generate_length=500):
    """Génère des notes à partir du modèle entraîné."""
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    # Choisir un point de départ aléatoire
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    # Générer notes
    for note_index in range(generate_length):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern = np.vstack((pattern[1:], [[index / float(n_vocab)]]))

    return prediction_output

# Fonction pour convertir la sortie en MIDI
def create_midi(prediction_output, output_file='output.mid'):
    """Convertit la sortie de prédiction en un fichier MIDI."""
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        # Accord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.duration.quarterLength = 0.5
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.duration.quarterLength = 0.5
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Incrémenter l'offset pour que les notes ne se chevauchent pas
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)
    print(f"Fichier MIDI généré: {output_file}")

# Fonction pour améliorer une mélodie existante
def enhance_melody(simple_midi_file, model_file='model.h5', output_file='enhanced_output.mid'):
    """Améliore une mélodie MIDI simple en utilisant le modèle entraîné."""
    # Charger le modèle
    model = load_model(model_file)

    # Charger les notes originales
    with open('notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)
    n_vocab = len(set(notes))
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # Préparer la séquence d'entrée à partir du MIDI simple
    midi = converter.parse(simple_midi_file)
    notes_to_parse = None
    parts = instrument.partitionByInstrument(midi)
    if parts:
        notes_to_parse = parts.parts[0].recurse()
    else:
        notes_to_parse = midi.flat.notes

    input_notes = []
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            input_notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            input_notes.append('.'.join(str(n) for n in element.normalOrder))

    # Convertir les notes en entiers
    input_sequence = [note_to_int[char] for char in input_notes if char in note_to_int]
    # S'assurer que la séquence est de la bonne longueur
    sequence_length = 100
    if len(input_sequence) < sequence_length:
        # Remplir avec des zéros ou répéter le motif
        input_sequence = [0]*(sequence_length - len(input_sequence)) + input_sequence
    else:
        input_sequence = input_sequence[:sequence_length]

    pattern = np.array(input_sequence) / float(n_vocab)
    pattern = np.reshape(pattern, (sequence_length, 1))
    prediction_output = []

    # Générer la mélodie améliorée
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = pitchnames[index]
        prediction_output.append(result)

        pattern = np.vstack((pattern[1:], [[index / float(n_vocab)]]))

    # Convertir la sortie en MIDI
    create_midi(prediction_output, output_file=output_file)
    print(f"Mélodie améliorée enregistrée dans {output_file}")

# Programme principal
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Modèle LSTM pour la génération de mélodies MIDI.')
    parser.add_argument('--collect', action='store_true', help='Collecter les fichiers MIDI dans le répertoire data.')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle LSTM.')
    parser.add_argument('--generate', action='store_true', help='Générer une nouvelle mélodie.')
    parser.add_argument('--enhance', type=str, help='Améliorer une mélodie MIDI existante.')
    parser.add_argument('--source_dir', type=str, default='.', help='Répertoire source pour collecter les fichiers MIDI.')
    parser.add_argument('--output_file', type=str, default='output.mid', help='Nom du fichier MIDI de sortie.')

    args = parser.parse_args()

    if args.collect:
        collect_mid_files(args.source_dir)
    elif args.train:
        train_network()
    elif args.generate:
        # Charger le modèle
        model = load_model('model.h5')
        with open('notes.pkl', 'rb') as filepath:
            notes = pickle.load(filepath)
        n_vocab = len(set(notes))
        pitchnames = sorted(set(item for item in notes))
        network_input, _ = prepare_sequences(notes, n_vocab)
        prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
        create_midi(prediction_output, output_file=args.output_file)
    elif args.enhance:
        enhance_melody(args.enhance, output_file=args.output_file)
    else:
        print("Aucune action spécifiée. Utilisez --help pour afficher les options disponibles.")
