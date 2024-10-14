import sounddevice as sd
import numpy as np
import mido
from scipy.io.wavfile import write, read
import time
import tkinter as tk
from tkinter import messagebox
import librosa
import pretty_midi
from monophonic import wave_to_midi



def generate_metronome_sound(frequency=1000, duration=100):
    sample_rate = 44100
    t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), endpoint=False)
    sound = 0.5 * np.sin(2 * np.pi * frequency * t)
    sound = (sound * 32767).astype(np.int16)
    return sound, sample_rate

def record_voice(bpm, measures=4, beats_per_measure=4):
    fs = 44100
    beat_duration = 60 / bpm
    measure_duration = beats_per_measure * beat_duration
    total_duration = measures * measure_duration
    metronome_sound, _ = generate_metronome_sound()

    print("Chantez maintenant !")
    
    # Jouer le métronome pour chaque mesure
    for measure in range(measures):
        for beat in range(beats_per_measure):
            sd.play(metronome_sound, samplerate=fs)
            time.sleep(beat_duration)

    print(f"Enregistrement... ({total_duration:.1f} secondes)")
    recording = sd.rec(int(total_duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Enregistrement terminé.")

    write('voice_recording.wav', fs, recording)

    return 'voice_recording.wav'

def play_audio(file_path):
    fs, data = read(file_path)
    sd.play(data, fs)
    sd.wait()
    print("Lecture terminée.")


def audio_to_midi(audio_file):
    # Load the audio
    y, sr = librosa.load(audio_file)

    # Extract the notes (pianoroll) and bpm
    notes, estimated_bpm = wave_to_midi(y, sr)

    # Create a PrettyMIDI object
    midi = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)

    # Add the notes to the instrument
    for note in notes:
        note_start = note[0]
        note_end = note[1]
        note_pitch = int(note[2])  # Ensure pitch is an integer
        note_velocity = 100  # Set a default velocity

        midi_note = pretty_midi.Note(
            velocity=note_velocity,
            pitch=note_pitch,
            start=note_start,
            end=note_end
        )

        piano.notes.append(midi_note)

    # Add the instrument to the MIDI object
    midi.instruments.append(piano)

    # Save the MIDI file
    midi.write('output.mid')

    return 'output.mid'

    

def ask_to_convert():
    response = messagebox.askyesno("Conversion", "Voulez-vous convertir l'enregistrement en MIDI ?")
    if response:
        bpm = int(bpm_entry.get())
        audio_file = 'voice_recording.wav'
        audio_to_midi(audio_file)
        messagebox.showinfo("Info", "Fichier MIDI généré : output.mid")

def start_recording():
    try:
        bpm = int(bpm_entry.get())
        if bpm <= 0:
            raise ValueError("Le BPM doit être supérieur à zéro.")
        
        recording_label.config(text="Enregistrement en cours...", fg="green")
        root.update()

        audio_file = record_voice(bpm)
        recording_label.config(text="Enregistrement terminé.", fg="blue")

        play_button.config(state=tk.NORMAL)

    except ValueError as e:
        messagebox.showerror("Erreur", str(e))
    except Exception as e:
        messagebox.showerror("Erreur", "Une erreur est survenue : " + str(e))

def play_recorded_audio():
    play_audio('voice_recording.wav')
    ask_to_convert()

# Création de l'interface graphique
root = tk.Tk()
root.title("Enregistreur de Voix avec Métronome")

# Label pour le BPM
bpm_label = tk.Label(root, text="Entrez le BPM (Battements par minute) :")
bpm_label.pack(pady=10)

# Entry pour le BPM
bpm_entry = tk.Entry(root)
bpm_entry.pack(pady=10)

# Bouton pour commencer l'enregistrement
start_button = tk.Button(root, text="Commencer l'enregistrement", command=start_recording)
start_button.pack(pady=20)

# Bouton pour lire l'audio
play_button = tk.Button(root, text="Réécouter l'enregistrement", command=play_recorded_audio, state=tk.DISABLED)
play_button.pack(pady=20)

# Label pour l'indicateur d'enregistrement
recording_label = tk.Label(root, text="")
recording_label.pack(pady=10)

# Lancer l'interface
root.mainloop()
