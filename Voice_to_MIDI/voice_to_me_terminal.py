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
import torch
from midi_generator import load_model_from_huggingface, generate_continuation_midi


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
    midi.write('output.mid')
    print("Conversion audio en MIDI terminée.")

    return 'output.mid'

def improve_midi_with_ai(midi_file, model_name="skytnt/midi-model-tv2o-medium",max_len=256, temp=0.90, top_p=0.98, top_k=20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model_from_huggingface(model_name, device=device)
    
    output_midi_file = 'enhanced_output.mid'
    generate_continuation_midi(
        model=model,
        tokenizer=tokenizer,
        output_path=output_midi_file,
        input_midi_path=midi_file,
        max_len=max_len,
        temp=temp,
        top_p=top_p,
        top_k=top_k,
    )
    print(f"MIDI amélioré et sauvegardé dans {output_midi_file}")
    return output_midi_file

def ask_to_convert():
    response = messagebox.askyesno("Conversion", "Voulez-vous convertir l'enregistrement en MIDI ?")
    if response:
        bpm = int(bpm_entry.get())
        audio_file = 'voice_recording.wav'
        midi_file = audio_to_midi(audio_file)
        messagebox.showinfo("Info", "Fichier MIDI généré : output.mid")
        
        # Appel de l'amélioration IA
        model_name = "skytnt/midi-model-tv2o-medium"
          # Remplacez par le nom du modèle sur Hugging Face
        improved_midi = improve_midi_with_ai(midi_file, model_name)
        messagebox.showinfo("Info", f"MIDI amélioré généré : {improved_midi}")

def start_recording(bpm_entry, recording_label, play_button):
    try:
        bpm = int(bpm_entry.get())
        if bpm <= 0:
            raise ValueError("Le BPM doit être supérieur à zéro.")
        
        recording_label.config(text="Enregistrement en cours...", fg="green")

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

if __name__ == "__main__":
    try:
        bpm = int(input("Entrez le BPM : "))
        if bpm <= 0:
            raise ValueError("Le BPM doit être supérieur à zéro.")
        
        print("Enregistrement en cours...")
        audio_file = record_voice(bpm)
        print("Enregistrement terminé.")
        
        play_audio(audio_file)
        response = input("Voulez-vous convertir l'enregistrement en MIDI ? (y/n): ").lower()
        if response == 'y':

            midi_file = audio_to_midi(audio_file)
            print(f"Fichier MIDI généré : {midi_file}")
            
            model_name = "skytnt/midi-model-tv2o-medium"
            max_len = int(input("Entrez la longueur maximale du MIDI amélioré (256 par défaut) : "))
            temp = float(input("Entrez la température (0.90 par défaut) : ")) 
            top_p = float(input("Entrez le top_p (0.98 par défaut) : ")) 
            top_k =  int(input("Entrez le top_k (20 par défaut) : "))

            improved_midi = improve_midi_with_ai(midi_file, model_name, max_len, temp, top_p, top_k)
            print(f"MIDI amélioré généré : {improved_midi}")
    except ValueError as e:
        print(f"Erreur : {e}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")
