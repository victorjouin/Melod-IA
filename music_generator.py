import os
import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream, key as m21key
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

# Définir les durées acceptables globalement
acceptable_durations = [0.25, 0.5, 1.0, 2.0, 4.0]

def get_notes_and_chords(data_dir='data'):
    """Extrait les notes, les durées et les accords des fichiers MIDI dans un répertoire donné."""
    notes = []
    durations = []
    chords_list = []

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

            prev_chord = None
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                    durations.append(element.duration.quarterLength)
                    if prev_chord is not None:
                        chords_list.append(prev_chord)
                    else:
                        chords_list.append('N')  # 'N' pour No Chord
                    prev_chord = None
                elif isinstance(element, chord.Chord):
                    chord_str = '.'.join(str(p) for p in element.pitches)
                    notes.append(chord_str)
                    durations.append(element.duration.quarterLength)
                    prev_chord = chord_str
                    chords_list.append(prev_chord)
                else:
                    # Si l'élément n'est ni une note ni un accord, on garde le dernier accord
                    if prev_chord is not None:
                        chords_list.append(prev_chord)
                    else:
                        chords_list.append('N')

    # Sauvegarder les notes, les durées et les accords
    with open('notes.pkl', 'wb') as filepath:
        pickle.dump(notes, filepath)
    with open('durations.pkl', 'wb') as filepath:
        pickle.dump(durations, filepath)
    with open('chords.pkl', 'wb') as filepath:
        pickle.dump(chords_list, filepath)

    return notes, durations, chords_list

def prepare_sequences(notes, durations, chords_list, pitchnames, chordnames, durationnames, note_to_int, chord_to_int, duration_to_int, sequence_length=100):
    """Prépare les séquences pour l'entraînement du modèle Transformer."""
    network_input_notes = []
    network_input_durations = []
    network_input_chords = []
    network_output_notes = []
    network_output_durations = []
    network_output_chords = []

    for i in range(len(notes) - sequence_length):
        sequence_in_notes = notes[i:i + sequence_length]
        sequence_in_durations = durations[i:i + sequence_length]
        sequence_in_chords = chords_list[i:i + sequence_length]
        sequence_out_note = notes[i + sequence_length]
        sequence_out_duration = durations[i + sequence_length]
        sequence_out_chord = chords_list[i + sequence_length]

        input_sequence_notes = [note_to_int.get(note, 0) for note in sequence_in_notes]
        input_sequence_durations = [duration_to_int.get(duration, 0) for duration in sequence_in_durations]
        input_sequence_chords = [chord_to_int.get(chord, 0) for chord in sequence_in_chords]

        network_input_notes.append(input_sequence_notes)
        network_input_durations.append(input_sequence_durations)
        network_input_chords.append(input_sequence_chords)
        network_output_notes.append(note_to_int.get(sequence_out_note, 0))
        network_output_durations.append(duration_to_int.get(sequence_out_duration, 0))
        network_output_chords.append(chord_to_int.get(sequence_out_chord, 0))

    return (network_input_notes, network_input_durations, network_input_chords,
            network_output_notes, network_output_durations, network_output_chords)

def create_positional_encoding(seq_len, d_model):
    """Crée un encodage positionnel pour le Transformer."""
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates

    sines = np.sin(angle_rads[:, 0::2])
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

def transformer_encoder_layer(d_model, num_heads, dff, rate=0.1):
    """Crée une couche d'encodeur Transformer."""
    inputs = Input(shape=(None, d_model))
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attention = Dropout(rate)(attention)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attention)

    ffn = Dense(dff, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)

    return Model(inputs=inputs, outputs=out2)

def create_transformer_model(n_vocab_notes, n_vocab_durations, n_vocab_chords, max_seq_len, d_model=256, num_heads=8, dff=512, num_layers=4, rate=0.1):
    """Crée et compile le modèle Transformer."""
    note_inputs = Input(shape=(max_seq_len,), name='note_inputs', dtype=tf.int32)
    duration_inputs = Input(shape=(max_seq_len,), name='duration_inputs', dtype=tf.int32)
    chord_inputs = Input(shape=(max_seq_len,), name='chord_inputs', dtype=tf.int32)

    # Embedding pour les notes
    note_embedding = Embedding(input_dim=n_vocab_notes, output_dim=d_model)(note_inputs)
    # Embedding pour les durées
    duration_embedding = Embedding(input_dim=n_vocab_durations, output_dim=d_model)(duration_inputs)
    # Embedding pour les accords
    chord_embedding = Embedding(input_dim=n_vocab_chords, output_dim=d_model)(chord_inputs)

    # Somme des embeddings
    embeddings = note_embedding + duration_embedding + chord_embedding

    # Ajout de l'encodage positionnel
    pos_encoding = create_positional_encoding(max_seq_len, d_model)
    embeddings += pos_encoding

    inputs = [note_inputs, duration_inputs, chord_inputs]

    # Encodeur Transformer
    x = embeddings
    for _ in range(num_layers):
        x = transformer_encoder_layer(d_model, num_heads, dff, rate)(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate)(x)

    # Sortie pour les notes
    notes_out = Dense(n_vocab_notes, activation='softmax', name='notes')(x)
    # Sortie pour les durées (classification)
    durations_out = Dense(n_vocab_durations, activation='softmax', name='durations')(x)
    # Sortie pour les accords
    chords_out = Dense(n_vocab_chords, activation='softmax', name='chords')(x)

    model = Model(inputs=inputs, outputs=[notes_out, durations_out, chords_out])
    model.compile(loss={'notes': 'sparse_categorical_crossentropy',
                        'durations': 'sparse_categorical_crossentropy',
                        'chords': 'sparse_categorical_crossentropy'},
                  optimizer='adam')
    return model

def train_network():
    """Entraîne le modèle Transformer sur les données extraites avec les durées et les accords."""
    notes, durations_raw, chords_list = get_notes_and_chords()
    pitchnames = sorted(set(notes))
    chordnames = sorted(set(chords_list))

    # Convertir les durées en catégories
    durations = [min(acceptable_durations, key=lambda x: abs(x - d)) for d in durations_raw]
    durationnames = sorted(set(acceptable_durations))

    n_vocab_notes = len(pitchnames)
    n_vocab_durations = len(durationnames)
    n_vocab_chords = len(chordnames)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    duration_to_int = dict((duration, number) for number, duration in enumerate(durationnames))
    chord_to_int = dict((chord, number) for number, chord in enumerate(chordnames))

    sequence_length = 100
    (network_input_notes, network_input_durations, network_input_chords,
     network_output_notes, network_output_durations, network_output_chords) = prepare_sequences(
        notes, durations, chords_list, pitchnames, chordnames, durationnames, note_to_int, chord_to_int, duration_to_int, sequence_length)

    model = create_transformer_model(
        n_vocab_notes=n_vocab_notes,
        n_vocab_durations=n_vocab_durations,
        n_vocab_chords=n_vocab_chords,
        max_seq_len=sequence_length
    )

    inputs = {
        'note_inputs': np.array(network_input_notes),
        'duration_inputs': np.array(network_input_durations),
        'chord_inputs': np.array(network_input_chords)
    }

    outputs = {
        'notes': np.array(network_output_notes),
        'durations': np.array(network_output_durations),
        'chords': np.array(network_output_chords)
    }

    # Entraînement du modèle
    model.fit(
        inputs,
        outputs,
        epochs=50,
        batch_size=16
    )

    # Sauvegarder le modèle complet
    model.save('model.h5')

    # Sauvegarder les mappings
    with open('note_to_int.pkl', 'wb') as f:
        pickle.dump(note_to_int, f)
    with open('int_to_note.pkl', 'wb') as f:
        pickle.dump({v: k for k, v in note_to_int.items()}, f)

    with open('duration_to_int.pkl', 'wb') as f:
        pickle.dump(duration_to_int, f)
    with open('int_to_duration.pkl', 'wb') as f:
        pickle.dump({v: k for k, v in duration_to_int.items()}, f)

    with open('chord_to_int.pkl', 'wb') as f:
        pickle.dump(chord_to_int, f)
    with open('int_to_chord.pkl', 'wb') as f:
        pickle.dump({v: k for k, v in chord_to_int.items()}, f)

def generate_music(temperature=1.0, output_file='output.mid', num_measures=8, theory_weight=1.0):
    """Génère une mélodie avec accompagnement d'accords en utilisant le modèle entraîné."""
    # Charger les mappings
    with open('int_to_note.pkl', 'rb') as f:
        int_to_note = pickle.load(f)
    with open('int_to_duration.pkl', 'rb') as f:
        int_to_duration = pickle.load(f)
    with open('int_to_chord.pkl', 'rb') as f:
        int_to_chord = pickle.load(f)
    pitchnames = list(int_to_note.values())
    durationnames = list(int_to_duration.values())
    chordnames = list(int_to_chord.values())
    n_vocab_notes = len(pitchnames)
    n_vocab_durations = len(durationnames)
    n_vocab_chords = len(chordnames)

    # Charger le modèle
    model = load_model('model.h5')

    # Charger les mappings inverses
    with open('note_to_int.pkl', 'rb') as f:
        note_to_int = pickle.load(f)
    with open('duration_to_int.pkl', 'rb') as f:
        duration_to_int = pickle.load(f)
    with open('chord_to_int.pkl', 'rb') as f:
        chord_to_int = pickle.load(f)

    # Charger les données d'entraînement
    with open('notes.pkl', 'rb') as f:
        notes = pickle.load(f)
    with open('durations.pkl', 'rb') as f:
        durations_raw = pickle.load(f)
    with open('chords.pkl', 'rb') as f:
        chords_list = pickle.load(f)

    durations = [min(acceptable_durations, key=lambda x: abs(x - d)) for d in durations_raw]

    sequence_length = 100
    (network_input_notes, network_input_durations, network_input_chords,
     _, _, _) = prepare_sequences(
        notes, durations, chords_list, pitchnames, chordnames, durationnames, note_to_int, chord_to_int, duration_to_int, sequence_length)

    # Initialisation aléatoire
    start = np.random.randint(0, len(network_input_notes) - 1)
    pattern_notes = network_input_notes[start]
    pattern_durations = network_input_durations[start]
    pattern_chords = network_input_chords[start]

    prediction_output_notes = []
    prediction_output_durations = []
    prediction_output_chords = []

    total_duration = 0.0
    measure_duration = 4.0  # Signature rythmique 4/4
    desired_duration = num_measures * measure_duration

    while total_duration < desired_duration:
        inputs = {
            'note_inputs': np.array([pattern_notes], dtype=np.int32),
            'duration_inputs': np.array([pattern_durations], dtype=np.int32),
            'chord_inputs': np.array([pattern_chords], dtype=np.int32)
        }

        prediction = model.predict(inputs, verbose=0)
        prediction_notes = prediction[0][0]
        prediction_durations = prediction[1][0]
        prediction_chords = prediction[2][0]

        # Application de la température pour les notes
        prediction_notes = np.log(prediction_notes + 1e-9) / temperature
        exp_preds_notes = np.exp(prediction_notes)
        prediction_notes = exp_preds_notes / np.sum(exp_preds_notes)

        # Application de la température pour les durées
        prediction_durations = np.log(prediction_durations + 1e-9) / temperature
        exp_preds_durations = np.exp(prediction_durations)
        prediction_durations = exp_preds_durations / np.sum(exp_preds_durations)

        # Application de la température pour les accords
        prediction_chords = np.log(prediction_chords + 1e-9) / temperature
        exp_preds_chords = np.exp(prediction_chords)
        prediction_chords = exp_preds_chords / np.sum(exp_preds_chords)

        # Sélection de l'index de la note
        possible_indices = [idx for idx in range(n_vocab_notes) if '.' not in int_to_note[idx]]
        prediction_notes_filtered = prediction_notes[possible_indices]
        prediction_notes_filtered /= np.sum(prediction_notes_filtered)
        index_note = np.random.choice(possible_indices, p=prediction_notes_filtered)
        result_note = int_to_note[index_note]

        # Sélection de l'index de la durée
        index_duration = np.random.choice(range(n_vocab_durations), p=prediction_durations)
        result_duration = int_to_duration[index_duration]

        # Sélection de l'index de l'accord
        index_chord = np.random.choice(range(n_vocab_chords), p=prediction_chords)
        result_chord = int_to_chord[index_chord]

        # Ajouter à la sortie
        prediction_output_notes.append((result_note, result_duration))
        prediction_output_chords.append((result_chord, result_duration))
        total_duration += result_duration

        # Mise à jour du pattern
        pattern_notes = np.append(pattern_notes[1:], index_note)
        pattern_durations = np.append(pattern_durations[1:], index_duration)
        pattern_chords = np.append(pattern_chords[1:], index_chord)

    # Créer le fichier MIDI
    create_full_midi(prediction_output_chords, prediction_output_notes, output_file=output_file)

def create_full_midi(chords_output, melody_output, output_file='output.mid'):
    """Crée un fichier MIDI combinant les accords et la mélodie."""
    offset = 0
    output_chords = []
    output_melody = []

    # Création de la partie des accords
    for chord_str, duration in chords_output:
        if chord_str == 'N':
            offset += duration
            continue
        notes_in_chord = chord_str.split('.')
        notes = []
        for note_str in notes_in_chord:
            try:
                new_note = note.Note(note_str)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            except:
                continue  # Ignorer les notes invalides
        if notes:
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            new_chord.quarterLength = duration
            output_chords.append(new_chord)
        offset += duration

    # Création de la partie de la mélodie
    offset = 0
    for note_str, duration in melody_output:
        if '.' in note_str:
            # Ignorer les accords dans la mélodie
            continue
        try:
            new_note = note.Note(note_str)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Violin()
            new_note.quarterLength = duration
            output_melody.append(new_note)
        except:
            pass  # Ignorer les notes invalides
        offset += duration

    # Combinaison des deux parties
    chords_part = stream.Part(output_chords)
    chords_part.insert(0, instrument.Piano())

    melody_part = stream.Part(output_melody)
    melody_part.insert(0, instrument.Violin())

    combined_score = stream.Score()
    combined_score.insert(0, chords_part)
    combined_score.insert(0, melody_part)

    combined_score.write('midi', fp=output_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Modèle Transformer pour la génération de mélodies avec accords.')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle Transformer.')
    parser.add_argument('--generate', action='store_true', help='Générer une nouvelle mélodie avec accords.')
    parser.add_argument('--output_file', type=str, default='output.mid', help='Nom du fichier MIDI de sortie.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Température pour la génération de notes.')
    parser.add_argument('--measures', type=int, default=8, help='Nombre de mesures à générer.')
    parser.add_argument('--theory_weight', type=float, default=1.0, help='Poids du respect de la théorie musicale.')

    args = parser.parse_args()

    if args.train:
        train_network()
    if args.generate:
        generate_music(
            temperature=args.temperature,
            output_file=args.output_file,
            num_measures=args.measures,
            theory_weight=args.theory_weight
        )
