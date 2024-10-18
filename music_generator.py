import os
import numpy as np
import pickle
from music21 import converter, instrument, note, chord, stream, key as m21key
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, MultiHeadAttention, LayerNormalization, Add, GlobalAveragePooling1D

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
    """Prépare les séquences pour l'entraînement du modèle Transformer."""
    network_input_notes = []
    network_input_durations = []
    network_output_notes = []
    network_output_durations = []

    for i in range(len(notes) - sequence_length):
        sequence_in_notes = notes[i:i + sequence_length]
        sequence_in_durations = durations[i:i + sequence_length]
        sequence_out_note = notes[i + sequence_length]
        sequence_out_duration = durations[i + sequence_length]

        input_sequence_notes = [note_to_int[note] for note in sequence_in_notes]
        input_sequence_durations = sequence_in_durations

        network_input_notes.append(input_sequence_notes)
        network_input_durations.append(input_sequence_durations)
        network_output_notes.append(note_to_int[sequence_out_note])
        network_output_durations.append(sequence_out_duration)

    max_duration = np.max(durations)

    # Conversion en tableaux NumPy avec types de données explicites
    network_input_notes = np.array(network_input_notes, dtype=np.int32)
    network_input_durations = np.array(network_input_durations, dtype=np.float32) / max_duration  # Normalisation
    network_output_notes = np.array(network_output_notes, dtype=np.int32)
    network_output_durations = np.array(network_output_durations, dtype=np.float32) / max_duration  # Normalisation

    # Vérification des formes et des types de données
    print(f"network_input_notes dtype: {network_input_notes.dtype}, shape: {network_input_notes.shape}")
    print(f"network_input_durations dtype: {network_input_durations.dtype}, shape: {network_input_durations.shape}")
    print(f"network_output_notes dtype: {network_output_notes.dtype}, shape: {network_output_notes.shape}")
    print(f"network_output_durations dtype: {network_output_durations.dtype}, shape: {network_output_durations.shape}")

    return network_input_notes, network_input_durations, network_output_notes, network_output_durations, max_duration

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

def create_transformer_model(n_vocab, max_seq_len, d_model=512, num_heads=12, dff=512, num_layers=32, rate=0.1, conditional=False):
    """Crée et compile le modèle Transformer."""
    note_inputs = Input(shape=(max_seq_len,), name='note_inputs', dtype=tf.int32)
    duration_inputs = Input(shape=(max_seq_len,), name='duration_inputs', dtype=tf.float32)

    # Embedding pour les notes
    note_embedding = Embedding(input_dim=n_vocab, output_dim=d_model)(note_inputs)
    # Embedding pour les durées
    duration_embedding = Dense(d_model)(tf.expand_dims(duration_inputs, -1))

    # Somme des embeddings
    embeddings = note_embedding + duration_embedding

    # Ajout de l'encodage positionnel
    pos_encoding = create_positional_encoding(max_seq_len, d_model)
    embeddings += pos_encoding

    # Encodeur Transformer
    x = embeddings
    for _ in range(num_layers):
        x = transformer_encoder_layer(d_model, num_heads, dff, rate)(x)

    # Si le modèle est conditionné par une mélodie, on ajoute une entrée supplémentaire
    if conditional:
        condition_inputs = Input(shape=(max_seq_len,), name='condition_inputs', dtype=tf.int32)
        condition_embedding = Embedding(input_dim=n_vocab, output_dim=d_model)(condition_inputs)
        x = Add()([x, condition_embedding])
        inputs = [note_inputs, duration_inputs, condition_inputs]
    else:
        inputs = [note_inputs, duration_inputs]

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(rate)(x)

    # Sortie pour les notes
    notes_out = Dense(n_vocab, activation='softmax', name='notes')(x)
    # Sortie pour les durées
    durations_out = Dense(1, activation='linear', name='durations')(x)

    model = Model(inputs=inputs, outputs=[notes_out, durations_out])
    model.compile(loss={'notes': 'sparse_categorical_crossentropy', 'durations': 'mean_squared_error'}, optimizer='adam')

    return model

def train_network(conditional=False):
    """Entraîne le modèle Transformer sur les données extraites avec les durées."""
    notes, durations = get_notes()
    pitchnames = sorted(set(notes))
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input_notes, network_input_durations, network_output_notes, network_output_durations, max_duration = prepare_sequences(
        notes, durations, pitchnames, note_to_int, sequence_length)

    # Vérification des formes
    print(f"Forme des entrées notes : {network_input_notes.shape}")
    print(f"Forme des entrées durées : {network_input_durations.shape}")
    print(f"Forme des sorties notes : {network_output_notes.shape}")
    print(f"Forme des sorties durées : {network_output_durations.shape}")

    model = create_transformer_model(
        n_vocab, 
        max_seq_len=sequence_length, 
        conditional=conditional
    )

    inputs = {
        'note_inputs': network_input_notes,
        'duration_inputs': network_input_durations
    }

    if conditional:
        # Pour simplifier, utilisons les mêmes entrées comme condition
        inputs['condition_inputs'] = network_input_notes

    # Entraînement du modèle
    model.fit(
        inputs,
        {'notes': network_output_notes, 'durations': network_output_durations},
        epochs=150,
        batch_size=16
    )

    # Sauvegarder le modèle et le max_duration
    model.save('model.h5')
    with open('max_duration.pkl', 'wb') as f:
        pickle.dump(max_duration, f)

    # Sauvegarder int_to_note pour la génération
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    with open('int_to_note.pkl', 'wb') as f:
        pickle.dump(int_to_note, f)

def generate_notes(model, pitchnames, int_to_note, max_duration, key='C', temperature=1.0, num_measures=8, conditional=False):
    """Génère une séquence de notes et de durées à partir du modèle entraîné."""
    from music21 import note as m21note

    # Obtenir la tonalité spécifiée
    key_signature = m21key.Key(key)
    allowed_pitches = [p.name for p in key_signature.pitches]

    n_vocab = len(pitchnames)

    # Charger les séquences d'entrée
    with open('notes.pkl', 'rb') as filepath:
        notes = pickle.load(filepath)
    with open('durations.pkl', 'rb') as filepath:
        durations = pickle.load(filepath)

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input_notes, network_input_durations, _, _, _ = prepare_sequences(
        notes, durations, pitchnames, note_to_int, sequence_length)

    start = np.random.randint(0, len(network_input_notes) - 1)
    pattern_notes = network_input_notes[start]
    pattern_durations = network_input_durations[start]

    prediction_output = []
    total_duration = 0.0
    measure_duration = 4.0  # Supposons une signature rythmique de 4/4

    desired_duration = num_measures * measure_duration

    while total_duration < desired_duration:
        inputs = {
            'note_inputs': np.array([pattern_notes], dtype=np.int32),
            'duration_inputs': np.array([pattern_durations], dtype=np.float32)
        }

        if conditional:
            # Pour simplifier, utilisons le pattern_notes comme condition
            inputs['condition_inputs'] = np.array([pattern_notes], dtype=np.int32)

        prediction = model.predict(inputs, verbose=0)
        prediction_notes = prediction[0][0]
        prediction_durations = prediction[1][0]

        # Application de la température pour les notes
        prediction_notes = np.log(prediction_notes + 1e-9) / temperature
        exp_preds = np.exp(prediction_notes)
        prediction_notes = exp_preds / np.sum(exp_preds)

        # Ajuster les probabilités en fonction de la tonalité
        mask = np.ones(n_vocab)
        for idx in range(n_vocab):
            note_str = int_to_note[idx]
            if '.' in note_str:
                # Accord
                notes_in_chord = note_str.split('.')
                in_key = all(m21note.Pitch(n).name in allowed_pitches for n in notes_in_chord)
            else:
                in_key = m21note.Pitch(note_str).name in allowed_pitches
            if not in_key:
                mask[idx] *= 0.1  # Réduire la probabilité pour les notes hors tonalité
        prediction_notes *= mask
        prediction_notes /= np.sum(prediction_notes)

        # Sélection de l'index de la note
        index = np.random.choice(range(n_vocab), p=prediction_notes)
        result_note = int_to_note[index]

        # Récupération de la durée prédite
        result_duration = prediction_durations * max_duration

        # Correction des durées négatives ou nulles
        if result_duration <= 0:
            result_duration = 0.25  # Valeur minimale par défaut

        # Ajouter à la sortie
        prediction_output.append((result_note, result_duration))
        total_duration += result_duration

        # Mise à jour du pattern
        pattern_notes = np.append(pattern_notes[1:], index)
        pattern_durations = np.append(pattern_durations[1:], result_duration / max_duration)

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

def generate_music(temperature=1.0, output_file='output.mid', key='C', num_measures=8, conditional=False):
    """Génère une nouvelle mélodie et l'enregistre dans un fichier MIDI."""
    with open('int_to_note.pkl', 'rb') as filepath:
        int_to_note = pickle.load(filepath)
    with open('max_duration.pkl', 'rb') as filepath:
        max_duration = pickle.load(filepath)
    pitchnames = list(int_to_note.values())
    n_vocab = len(pitchnames)

    model = load_model('model.h5', compile=False)
    model.compile(loss={'notes': 'sparse_categorical_crossentropy', 'durations': 'mean_squared_error'}, optimizer='adam')

    # Générer les notes
    prediction_output = generate_notes(model, pitchnames, int_to_note, max_duration, key=key, temperature=temperature, num_measures=num_measures, conditional=conditional)

    # Créer le fichier MIDI
    create_midi(prediction_output, output_file=output_file)

def enhance_midi(input_midi_file, output_midi_file='enhanced_output.mid', key='C', num_measures=8, temperature=1.0, conditional=False):
    """Améliore un fichier MIDI en y ajoutant une mélodie basée sur les notes existantes avec des durées variables."""
    # Charger le modèle et les données nécessaires
    with open('int_to_note.pkl', 'rb') as filepath:
        int_to_note = pickle.load(filepath)
    with open('max_duration.pkl', 'rb') as filepath:
        max_duration = pickle.load(filepath)
    pitchnames = list(int_to_note.values())
    n_vocab = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    # Extraire les notes et durées du fichier MIDI d'entrée
    input_notes, input_durations = get_notes_from_midi_with_durations(input_midi_file)

    # Vérifier si le fichier MIDI contient assez de notes
    if len(input_notes) < 100:
        print("Le fichier MIDI d'entrée ne contient pas assez de notes pour la condition.")
        return

    # Préparer les séquences pour la génération
    sequence_length = 100
    input_notes_seq = [note_to_int.get(n, 0) for n in input_notes]
    input_durations_seq = input_durations

    # Limiter la séquence à la longueur maximale
    input_notes_seq = input_notes_seq[:sequence_length]
    input_durations_seq = input_durations_seq[:sequence_length]
    input_durations_seq = np.array(input_durations_seq, dtype=np.float32) / max_duration

    # Charger le modèle
    model = load_model('model.h5', compile=False)
    model.compile(loss={'notes': 'sparse_categorical_crossentropy', 'durations': 'mean_squared_error'}, optimizer='adam')

    # Générer une nouvelle mélodie conditionnée sur la mélodie d'entrée
    prediction_output = []

    pattern_notes = np.array(input_notes_seq, dtype=np.int32)
    pattern_durations = np.array(input_durations_seq, dtype=np.float32)

    total_duration = 0.0
    measure_duration = 4.0
    desired_duration = num_measures * measure_duration

    while total_duration < desired_duration:
        inputs = {
            'note_inputs': np.array([pattern_notes], dtype=np.int32),
            'duration_inputs': np.array([pattern_durations], dtype=np.float32)
        }

        if conditional:
            inputs['condition_inputs'] = np.array([pattern_notes], dtype=np.int32)

        prediction = model.predict(inputs, verbose=0)
        prediction_notes = prediction[0][0]
        prediction_durations = prediction[1][0]

        # Application de la température pour les notes
        prediction_notes = np.log(prediction_notes + 1e-9) / temperature
        exp_preds = np.exp(prediction_notes)
        prediction_notes = exp_preds / np.sum(exp_preds)

        # Ajuster les probabilités en fonction de la tonalité
        from music21 import note as m21note, key as m21key
        key_signature = m21key.Key(key)
        allowed_pitches = [p.name for p in key_signature.pitches]

        mask = np.ones(n_vocab)
        for idx in range(n_vocab):
            note_str = int_to_note[idx]
            if '.' in note_str:
                notes_in_chord = note_str.split('.')
                in_key = all(m21note.Pitch(n).name in allowed_pitches for n in notes_in_chord)
            else:
                in_key = m21note.Pitch(note_str).name in allowed_pitches
            if not in_key:
                mask[idx] *= 0.1
        prediction_notes *= mask
        prediction_notes /= np.sum(prediction_notes)

        # Sélection de l'index de la note
        index = np.random.choice(range(n_vocab), p=prediction_notes)
        result_note = int_to_note[index]

        # Récupération de la durée prédite
        result_duration = prediction_durations * max_duration

        if result_duration <= 0:
            result_duration = 0.25

        prediction_output.append((result_note, result_duration))
        total_duration += result_duration

        # Mise à jour du pattern
        pattern_notes = np.append(pattern_notes[1:], index)
        pattern_durations = np.append(pattern_durations[1:], result_duration / max_duration)

    # Créer le flux MIDI de la nouvelle mélodie
    output_notes = []
    offset = 0

    acceptable_durations = [4.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25, 0.125, 0.0625]

    for pattern, duration in prediction_output:
        closest_duration = min(acceptable_durations, key=lambda x: abs(x - duration))
        if closest_duration <= 0:
            closest_duration = 0.25

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

    new_melody_stream = stream.Part(output_notes)
    new_melody_stream.insert(0, instrument.Violin())

    original_midi = converter.parse(input_midi_file)
    if not isinstance(original_midi, stream.Part):
        original_midi_part = stream.Part()
        original_midi_part.append(original_midi)
    else:
        original_midi_part = original_midi

    original_midi_part.insert(0, instrument.Piano())

    combined_score = stream.Score()
    combined_score.insert(0, original_midi_part)
    combined_score.insert(0, new_melody_stream)

    combined_score.makeMeasures(inPlace=True)

    combined_score.write('midi', fp=output_midi_file)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Modèle Transformer pour la génération et l\'amélioration de mélodies MIDI.')
    parser.add_argument('--train', action='store_true', help='Entraîner le modèle Transformer.')
    parser.add_argument('--generate', action='store_true', help='Générer une nouvelle mélodie.')
    parser.add_argument('--enhance', action='store_true', help='Améliorer un fichier MIDI existant.')
    parser.add_argument('--input_midi', type=str, help='Chemin du fichier MIDI d\'entrée pour l\'amélioration.')
    parser.add_argument('--output_file', type=str, default='output.mid', help='Nom du fichier MIDI de sortie.')
    parser.add_argument('--temperature', type=float, default=1.0, help='Température pour la génération de notes.')
    parser.add_argument('--key', type=str, default='C', help='Tonalité pour la génération de la musique.')
    parser.add_argument('--measures', type=int, default=8, help='Nombre de mesures à générer.')
    parser.add_argument('--conditional', action='store_true', help='Utiliser le modèle conditionné par une mélodie.')

    args = parser.parse_args()

    if args.train:
        train_network(conditional=args.conditional)
    if args.generate:
        generate_music(temperature=args.temperature, output_file=args.output_file, key=args.key, num_measures=args.measures, conditional=args.conditional)
    if args.enhance:
        if args.input_midi:
            enhance_midi(input_midi_file=args.input_midi, output_midi_file=args.output_file, key=args.key, num_measures=args.measures, temperature=args.temperature, conditional=args.conditional)
        else:
            print("Veuillez fournir le chemin du fichier MIDI d'entrée avec l'argument --input_midi.")
