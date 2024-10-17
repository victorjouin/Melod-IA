import os
import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import tensorflow as tf

def get_notes_from_midi(midi_file):
    """Extrait les notes et accords d'un fichier MIDI donné."""
    notes = []

    midi = converter.parse(midi_file)
    notes_to_parse = None

    try:
        s2 = instrument.partitionByInstrument(midi)
        parts = s2.parts
        notes_to_parse = parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            chord_str = '.'.join(str(p) for p in element.pitches)
            notes.append(chord_str)

    return notes

def get_notes(data_dir='data'):
    """Extrait les notes et leurs durées des fichiers MIDI dans un répertoire donné."""
    notes = []
    durations = []

    for file in os.listdir(data_dir):
        if file.lower().endswith(('.mid', '.midi')):
            file_path = os.path.join(data_dir, file)
            midi = converter.parse(file_path)

            notes_to_parse = None

            try:
                s2 = instrument.partitionByInstrument(midi)
                notes_to_parse = s2.parts[0].recurse()
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    durations.append(element.duration.quarterLength)
                elif isinstance(element, chord.Chord):
                    chord_str = '.'.join(str(p) for p in element.pitches)
                    notes.append(chord_str)
                    durations.append(element.duration.quarterLength)

    # Sauvegarder les notes et les durées
    with open('notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)
    with open('durations.pkl', 'wb') as filepath:
        pickle.dump(durations, filepath)

    return notes, durations

def prepare_sequences(notes, durations, pitchnames, note_to_int, sequence_length=100):
    """Prépare les séquences pour l'entraînement du modèle en incluant les durées."""
    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in_notes = notes[i:i + sequence_length]
        sequence_in_durations = durations[i:i + sequence_length]
        sequence_out_note = notes[i + sequence_length]
        sequence_out_duration = durations[i + sequence_length]

        input_sequence_notes = [note_to_int[note] for note in sequence_in_notes]
        input_sequence_durations = sequence_in_durations

        # Combiner les notes et les durées
        input_sequence = list(zip(input_sequence_notes, input_sequence_durations))
        output_sequence = (note_to_int[sequence_out_note], sequence_out_duration)

        network_input.append(input_sequence)
        network_output.append(output_sequence)

    n_patterns = len(network_input)

    # Préparation des données pour le modèle
    X_notes = np.array([[pair[0] for pair in seq] for seq in network_input], dtype=np.float32)
    X_durations = np.array([[pair[1] for pair in seq] for seq in network_input], dtype=np.float32)
    y_notes = np.array([output[0] for output in network_output], dtype=np.int32)
    y_durations = np.array([output[1] for output in network_output], dtype=np.float32)

    # Reshape des entrées
    X_notes = np.reshape(X_notes, (n_patterns, sequence_length, 1))
    X_notes = X_notes / float(len(pitchnames))  # Normalisation

    X_durations = np.reshape(X_durations, (n_patterns, sequence_length, 1))
    max_duration = np.max(X_durations)
    X_durations = X_durations / max_duration  # Normalisation

    # Combiner les notes et les durées en une seule entrée
    network_input = np.concatenate((X_notes, X_durations), axis=2)

    # Préparation des sorties
    y_notes = tf.keras.utils.to_categorical(y_notes, num_classes=len(pitchnames))

    # Normaliser y_durations et reshaper
    y_durations = y_durations / max_duration
    y_durations = y_durations.reshape(-1, 1).astype(np.float32)

    return network_input, y_notes, y_durations, max_duration

def create_network(network_input, n_vocab):
    """Crée et compile le modèle LSTM pour prédire les notes et les durées."""
    notes_in = Input(shape=(network_input.shape[1], 2))
    x = LSTM(512, return_sequences=True)(notes_in)
    x = Dropout(0.3)(x)
    x = LSTM(512)(x)
    x = Dense(256)(x)
    x = Dropout(0.3)(x)

    # Sortie pour les notes
    notes_out = Dense(n_vocab, activation='softmax', name='notes')(x)
    # Sortie pour les durées
    durations_out = Dense(1, activation='linear', name='durations')(x)

    model = Model(inputs=notes_in, outputs=[notes_out, durations_out])
    model.compile(loss={'notes': 'categorical_crossentropy', 'durations': 'mean_squared_error'}, optimizer='adam')

    return model

def train_network():
    """Entraîne le modèle sur les données extraites avec les durées."""
    notes, durations = get_notes()
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input, y_notes, y_durations, max_duration = prepare_sequences(notes, durations, pitchnames, note_to_int)

    # Vérification des types et formes
    print(f"Type de network_input: {network_input.dtype}, Forme: {network_input.shape}")
    print(f"Type de y_notes: {y_notes.dtype}, Forme: {y_notes.shape}")
    print(f"Type de y_durations: {y_durations.dtype}, Forme: {y_durations.shape}")

    model = create_network(network_input, n_vocab)

    # Entraînement du modèle
    model.fit(
        network_input,
        {'notes': y_notes, 'durations': y_durations},
        epochs=100,
        batch_size=64
    )

    # Sauvegarder le modèle et le max_duration
    model.save('model.h5')
    with open('max_duration.pkl', 'wb') as f:
        pickle.dump(max_duration, f)

    # Sauvegarder int_to_note pour la génération
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    with open('int_to_note.pkl', 'wb') as f:
        pickle.dump(int_to_note, f)

def generate_notes(model, network_input, pitchnames, int_to_note, max_duration, temperature=1.0, generate_length=500):
    """Génère une séquence de notes et de durées à partir du modèle entraîné."""
    n_vocab = len(pitchnames)
    start = np.random.randint(0, len(network_input) - 1)
    pattern = network_input[start]
    prediction_output = []

    for note_index in range(generate_length):
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], 2))
        prediction = model.predict(prediction_input, verbose=0)
        prediction_notes = prediction[0][0]
        prediction_durations = prediction[1][0]

        # Application de la température pour les notes
        prediction_notes = np.log(prediction_notes + 1e-9) / temperature
        exp_preds = np.exp(prediction_notes)
        prediction_notes = exp_preds / np.sum(exp_preds)

        # Sélection de l'index de la note
        index = np.random.choice(range(n_vocab), p=prediction_notes)
        result_note = int_to_note[index]

        # Récupération de la durée prédite
        result_duration = prediction_durations[0] * max_duration

        # Correction des durées négatives ou nulles
        if result_duration <= 0:
            result_duration = 0.25  # Valeur minimale par défaut

        # Ajouter à la sortie
        prediction_output.append((result_note, result_duration))

        # Mise à jour du pattern
        next_input = np.array([index / float(n_vocab), prediction_durations[0]], dtype=np.float32)
        pattern = np.vstack([pattern[1:], next_input])

    return prediction_output

def create_midi(prediction_output, output_file='output.mid'):
    """Convertit une séquence de notes et de durées en fichier MIDI."""
    offset = 0
    output_notes = []

    # Définir les durées acceptables
    acceptable_durations = [4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625]

    for pattern, duration in prediction_output:
        # Quantifier la durée à la valeur acceptable la plus proche
        closest_duration = min(acceptable_durations, key=lambda x: abs(x - duration))

        # S'assurer que la durée est positive
        if closest_duration <= 0:
            closest_duration = 0.25  # Valeur minimale par défaut

        # Créer la note ou l'accord
        if ('.' in pattern):
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.quarterLength = closest_duration
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.quarterLength = closest_duration
            output_notes.append(new_note)
        offset += closest_duration  # Incrémenter l'offset en fonction de la durée

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_file)

def get_notes_from_midi_with_durations(midi_file):
    """Extrait les notes et leurs durées d'un fichier MIDI donné."""
    notes = []
    durations = []

    midi = converter.parse(midi_file)
    notes_to_parse = None

    try:
        s2 = instrument.partitionByInstrument(midi)
        parts = s2.parts
        notes_to_parse = parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
            durations.append(element.duration.quarterLength)
        elif isinstance(element, chord.Chord):
            chord_str = '.'.join(str(p) for p in element.pitches)
            notes.append(chord_str)
            durations.append(element.duration.quarterLength)

    return notes, durations

def generate_music(generate_length=500, temperature=1.0, output_file='output.mid'):
    """Génère une nouvelle mélodie et l'enregistre dans un fichier MIDI."""
    with open('notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)
    with open('durations.pkl', 'rb') as filepath:
        durations = pickle.load(filepath)
    with open('int_to_note.pkl', 'rb') as filepath:
        int_to_note = pickle.load(filepath)
    with open('max_duration.pkl', 'rb') as filepath:
        max_duration = pickle.load(filepath)
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # Préparer les séquences
    sequence_length = 100
    network_input, _, _, _ = prepare_sequences(notes, durations, pitchnames, note_to_int, sequence_length)
    model = load_model('model.h5', compile=False)
    model.compile(loss={'notes': 'categorical_crossentropy', 'durations': 'mean_squared_error'}, optimizer='adam')

    # Générer les notes
    prediction_output = generate_notes(model, network_input, pitchnames, int_to_note, max_duration, temperature, generate_length)

    # Créer le fichier MIDI
    create_midi(prediction_output, output_file=output_file)

def enhance_midi(input_midi_file, output_midi_file='enhanced_output.mid', generate_length=500, temperature=1.0):
    """Améliore un fichier MIDI en y ajoutant une mélodie basée sur les notes existantes avec des durées variables."""
    # Charger le modèle et les données nécessaires
    with open('notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)
    with open('durations.pkl', 'rb') as filepath:
        durations = pickle.load(filepath)
    with open('int_to_note.pkl', 'rb') as filepath:
        int_to_note = pickle.load(filepath)
    with open('max_duration.pkl', 'rb') as filepath:
        max_duration = pickle.load(filepath)
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # Extraire les notes et durées du fichier MIDI d'entrée
    input_notes, input_durations = get_notes_from_midi_with_durations(input_midi_file)

    # Si le fichier MIDI ne contient pas assez de notes, utiliser les notes du jeu de données
    if len(input_notes) < 100:
        input_notes = notes
        input_durations = durations

    # Préparer les séquences pour la génération
    sequence_length = 100
    network_input, _, _, _ = prepare_sequences(input_notes, input_durations, pitchnames, note_to_int, sequence_length)

    model = load_model('model.h5', compile=False)
    model.compile(loss={'notes': 'categorical_crossentropy', 'durations': 'mean_squared_error'}, optimizer='adam')

    # Générer une nouvelle mélodie basée sur les notes du fichier MIDI d'entrée
    prediction_output = generate_notes(model, network_input, pitchnames, int_to_note, max_duration, temperature, generate_length)

    # Créer le flux MIDI de la nouvelle mélodie
    output_notes = []
    offset = 0

    # Définir les durées acceptables
    acceptable_durations = [4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625]

    for pattern, duration in prediction_output:
        # Quantifier la durée à la valeur acceptable la plus proche
        closest_duration = min(acceptable_durations, key=lambda x: abs(x - duration))

        # S'assurer que la durée est positive
        if closest_duration <= 0:
            closest_duration = 0.25  # Valeur minimale par défaut

        # Créer la note ou l'accord
        if ('.' in pattern):
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Violin()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.quarterLength = closest_duration
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Violin()
            new_note.quarterLength = closest_duration
            output_notes.append(new_note)
        offset += closest_duration

    # Créer un flux pour la nouvelle mélodie
    new_melody_stream = stream.Part(output_notes)
    new_melody_stream.insert(0, instrument.Violin())

    # Charger le fichier MIDI original
    original_midi = converter.parse(input_midi_file)

    # S'assurer que le MIDI original est dans un Part
    if not isinstance(original_midi, stream.Part):
        original_midi_part = stream.Part()
        original_midi_part.append(original_midi)
    else:
        original_midi_part = original_midi

    original_midi_part.insert(0, instrument.Piano())

    # Créer un Score et ajouter les parties
    combined_score = stream.Score()
    combined_score.insert(0, original_midi_part)
    combined_score.insert(0, new_melody_stream)

    # Ajouter des mesures au Score
    combined_score.makeMeasures(inPlace=True)

    # Sauvegarder le nouveau fichier MIDI
    combined_score.write('midi', fp=output_midi_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Modèle LSTM pour la génération et l\'amélioration de mélodies MIDI.')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle LSTM.')
    parser.add_argument('--generate', action='store_true', help='Générer une nouvelle mélodie.')
    parser.add_argument('--enhance', action='store_true', help='Améliorer un fichier MIDI existant.')
    parser.add_argument('--input_midi', type=str, help='Chemin du fichier MIDI d\'entrée pour l\'amélioration.')
    parser.add_argument('--output_file', type=str, default='output.mid', help='Nom du fichier MIDI de sortie.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Température pour la génération de notes.')
    parser.add_argument('--length', type=int, default=500, help='Longueur de la séquence générée.')

    args = parser.parse_args()

    if args.train:
        train_network()
    if args.generate:
        generate_music(generate_length=args.length, temperature=args.temperature, output_file=args.output_file)
    if args.enhance:
        if args.input_midi:
            enhance_midi(input_midi_file=args.input_midi, output_midi_file=args.output_file, generate_length=args.length, temperature=args.temperature)
        else:
            print("Veuillez fournir le chemin du fichier MIDI d'entrée avec l'argument --input_midi.")
