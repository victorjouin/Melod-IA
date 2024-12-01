import streamlit as st
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import pretty_midi
import torch
import time
import os
import subprocess
from midi_generator import load_model_from_huggingface, generate_continuation_midi
from monophonic import wave_to_midi

# --- Configuration de la page ---
st.set_page_config(
    page_title="MelodIA - Générateur de Musique Assisté par IA",
    page_icon="🎵",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Définition du répertoire de stockage des fichiers ---
DATA_DIR = "data"

# Créer le répertoire s'il n'existe pas
os.makedirs(DATA_DIR, exist_ok=True)

# Chemins des fichiers
METRONOME_FILE = os.path.join(DATA_DIR, 'metronome_track.wav')
VOICE_RECORDING_FILE = os.path.join(DATA_DIR, 'voice_recording.wav')
OUTPUT_MIDI_FILE = os.path.join(DATA_DIR, 'output.mid')
ENHANCED_MIDI_FILE = os.path.join(DATA_DIR, 'enhanced_output.mid')
OUTPUT_MP3_FILE = os.path.join(DATA_DIR, 'enhanced_output.mp3')

# --- Fonctions Utilitaires ---

# Fonction pour générer le son du métronome
def generate_metronome_sound(frequency=1000, duration=100, fs=44100):
    t = np.linspace(0, duration / 1000, int(fs * duration / 1000), endpoint=False)
    sound = 0.5 * np.sin(2 * np.pi * frequency * t)
    return sound

# Fonction pour générer la piste du métronome
def generate_metronome_track(bpm, measures=4, beats_per_measure=4, fs=44100):
    beat_duration = 60 / bpm
    total_beats = measures * beats_per_measure
    metronome_click = generate_metronome_sound(frequency=1000, duration=100, fs=fs)
    silence_duration = beat_duration - (len(metronome_click) / fs)
    silence = np.zeros(int(fs * silence_duration))
    metronome_track = np.array([])
    for _ in range(int(total_beats)):
        beat = np.concatenate((metronome_click, silence))
        metronome_track = np.concatenate((metronome_track, beat))
    metronome_track = (metronome_track * 32767).astype(np.int16)
    write(METRONOME_FILE, fs, metronome_track)
    return metronome_track, fs

# Fonction pour enregistrer l'audio
def record_voice(duration, fs=44100):
    st.info("**Enregistrement en cours...**")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(VOICE_RECORDING_FILE, fs, recording)
    st.success("**Enregistrement terminé.**")

# Fonction pour convertir l'audio en MIDI
def audio_to_midi(audio_file):
    y, sr = librosa.load(audio_file)
    notes, estimated_bpm = wave_to_midi(y, sr)
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    for note in notes:
        note_start = note[0]
        note_end = note[1]
        note_pitch = int(note[2])
        note_velocity = 100

        midi_note = pretty_midi.Note(
            velocity=note_velocity,
            pitch=note_pitch,
            start=note_start,
            end=note_end
        )

        piano.notes.append(midi_note)

    midi.instruments.append(piano)
    midi.write(OUTPUT_MIDI_FILE)
    st.success("**Conversion audio en MIDI terminée.**")

    return OUTPUT_MIDI_FILE

# Fonction pour améliorer le MIDI avec l'IA
def improve_midi_with_ai(midi_file, model_name="skytnt/midi-model-tv2o-medium"):
    st.info("**Amélioration du MIDI avec l'IA en cours...**")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_from_huggingface(model_name, device=device)

    output_midi_file = ENHANCED_MIDI_FILE
    generate_continuation_midi(
        model=model,
        tokenizer=tokenizer,
        output_path=output_midi_file,
        input_midi_path=midi_file,
        max_len=512,
        temp=0.90,
        top_p=0.98,
        top_k=20
    )
    st.success(f"**MIDI amélioré et sauvegardé dans {output_midi_file}**")
    return output_midi_file

# Fonction pour convertir le MIDI en MP3 en utilisant TiMidity++ et ffmpeg
def midi_to_mp3(midi_file, output_audio_file=OUTPUT_MP3_FILE):
    # Convertir le MIDI en WAV en utilisant TiMidity++
    wav_file = os.path.join(DATA_DIR, 'temp_output.wav')
    try:
        # Vérifier si TiMidity++ est installé
        subprocess.run(['timidity', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        st.error("TiMidity++ n'est pas installé. Veuillez l'installer pour convertir le MIDI en audio.")
        return None
    except FileNotFoundError:
        st.error("TiMidity++ n'est pas trouvé. Assurez-vous qu'il est installé et accessible depuis le PATH.")
        return None

    # Convertir MIDI en WAV
    try:
        subprocess.run(['timidity', midi_file, '-Ow', '-o', wav_file], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Erreur lors de la conversion du MIDI en WAV : {e}")
        return None

    # Convertir le WAV en MP3 en utilisant ffmpeg
    try:
        # Vérifier si ffmpeg est installé
        subprocess.run(['ffmpeg', '-version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        st.error("ffmpeg n'est pas installé. Veuillez l'installer pour convertir le WAV en MP3.")
        return None
    except FileNotFoundError:
        st.error("ffmpeg n'est pas trouvé. Assurez-vous qu'il est installé et accessible depuis le PATH.")
        return None

    try:
        subprocess.run(['ffmpeg', '-y', '-i', wav_file, output_audio_file], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Erreur lors de la conversion du WAV en MP3 : {e}")
        return None
    finally:
        # Supprimer le fichier WAV temporaire
        if os.path.exists(wav_file):
            os.remove(wav_file)

    st.success(f"**Conversion du MIDI en MP3 terminée : {output_audio_file}**")
    return output_audio_file

# --- Interface Principale ---

def main():
    # En-tête et description
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>🎵 MelodIA 🎵</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Transformez votre voix en mélodie assistée par IA</h3>", unsafe_allow_html=True)
    st.write("---")

    if 'step' not in st.session_state:
        st.session_state.step = 1

    # Étape 1 : Paramètres du métronome
    if st.session_state.step == 1:
        st.header("1️⃣ Paramètres du Métronome")
        bpm = st.number_input('**Entrez le BPM (Battements par minute) :**', min_value=1, max_value=300, value=120)
        measures = st.number_input('**Nombre de mesures :**', min_value=1, max_value=16, value=4)
        beats_per_measure = 4  # Vous pouvez rendre ceci modifiable si nécessaire

        total_duration = measures * beats_per_measure * (60 / bpm)

        if st.button("🎵 Jouer le Métronome"):
            metronome_track, fs = generate_metronome_track(bpm, measures, beats_per_measure)
            st.audio(METRONOME_FILE, format='audio/wav')
            st.success("Le métronome a fini de jouer. Vous pouvez maintenant enregistrer votre voix.")
            st.session_state.step = 2
            st.session_state.total_duration = total_duration

    # Étape 2 : Enregistrement de la voix
    if st.session_state.step == 2:
        st.header("2️⃣ Téléchargement de Votre Fichier Audio")
        st.write("Veuillez télécharger un fichier audio au format WAV.")
        uploaded_file = st.file_uploader("Choisissez un fichier", type=["wav"])

        if uploaded_file is not None:
            # Sauvegarder le fichier téléchargé
            with open(VOICE_RECORDING_FILE, 'wb') as f:
                f.write(uploaded_file.read())
            st.audio(VOICE_RECORDING_FILE, format='audio/wav')
            st.session_state.step = 3
    # Étape 3 : Conversion en MIDI
    if st.session_state.step == 3:
        st.header("3️⃣ Conversion de l'Enregistrement en MIDI")
        if st.button("🎼 Convertir en MIDI"):
            if os.path.exists(VOICE_RECORDING_FILE):
                midi_file = audio_to_midi(VOICE_RECORDING_FILE)
                with open(OUTPUT_MIDI_FILE, 'rb') as f:
                    st.download_button('💾 Télécharger le Fichier MIDI', f, file_name='output.mid')
                st.session_state.step = 4
            else:
                st.warning("Aucun enregistrement audio trouvé. Veuillez d'abord enregistrer votre voix.")

    # Étape 4 : Amélioration du MIDI avec l'IA
    if st.session_state.step == 4:
        st.header("4️⃣ Amélioration du MIDI avec l'IA")
        if st.button("🚀 Améliorer avec l'IA"):
            if os.path.exists(OUTPUT_MIDI_FILE):
                model_name = "skytnt/midi-model-tv2o-medium"
                improved_midi = improve_midi_with_ai(OUTPUT_MIDI_FILE, model_name)
                if improved_midi and os.path.exists(improved_midi):
                    with open(improved_midi, 'rb') as f:
                        st.download_button('💾 Télécharger le MIDI Amélioré', f, file_name='enhanced_output.mid')
                    st.session_state.step = 5
                else:
                    st.error("Échec de l'amélioration du MIDI.")
            else:
                st.warning("Aucun fichier MIDI trouvé. Veuillez d'abord convertir l'enregistrement en MIDI.")

    # Étape 5 : Conversion du MIDI en MP3
    if st.session_state.step == 5:
        st.header("5️⃣ Conversion du MIDI en MP3")
        if st.button("🎧 Convertir en MP3"):
            if os.path.exists(ENHANCED_MIDI_FILE):
                mp3_file = midi_to_mp3(ENHANCED_MIDI_FILE, OUTPUT_MP3_FILE)
                if mp3_file and os.path.exists(mp3_file):
                    st.audio(mp3_file, format='audio/mp3')
                    with open(mp3_file, 'rb') as f:
                        st.download_button('💾 Télécharger le Fichier MP3', f, file_name='enhanced_output.mp3')
                else:
                    st.error("La conversion du MIDI en MP3 a échoué.")
            else:
                st.warning("Aucun fichier MIDI amélioré trouvé. Veuillez d'abord améliorer le MIDI avec l'IA.")


if __name__ == "__main__":
    main()
