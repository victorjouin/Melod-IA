import argparse
import json
import os
from pathlib import Path
from typing import Union, Optional, List

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as safe_load_file

from midi_model import MIDIModel, MIDIModelConfig
from midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2, MIDITokenizer
from midi_synthesizer import MidiSynthesizer
import MIDI  # Assurez-vous que MIDI.py contient les fonctions midi2score et score2midi

# Définition des mappings
number2drum_kits = {
    -1: "None", 0: "Standard", 8: "Room", 16: "Power",
    24: "Electric", 25: "TR-808", 32: "Jazz",
    40: "Blush", 48: "Orchestra"
}
patch2number = {v: k for k, v in MIDI.Number2patch.items()}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}
key_signatures = [
    'C♭', 'A♭m', 'G♭', 'E♭m', 'D♭', 'B♭m', 'A♭', 'Fm', 'E♭', 'Cm', 'B♭', 'Gm',
    'F', 'Dm', 'C', 'Am', 'G', 'Em', 'D', 'Bm', 'A', 'F♯m', 'E', 'C♯m', 'B',
    'G♯m', 'F♯', 'D♯m', 'C♯', 'A♯m'
]

MAX_SEED = np.iinfo(np.int32).max

def load_model_from_huggingface(model_name: str, lora_path: Optional[str] = None, device: str = "cuda") -> (MIDIModel, MIDITokenizer):
    """
    Charge le modèle MIDI depuis Hugging Face.

    Args:
        model_name (str): Nom du modèle sur Hugging Face.
        lora_path (Optional[str], optional): Chemin vers les poids LoRA. Defaults to None.
        device (str, optional): Dispositif pour exécuter le modèle ('cuda' ou 'cpu'). Defaults to "cuda".

    Returns:
        MIDIModel: Modèle MIDI chargé.
        MIDITokenizer: Tokenizer associé.
    """
    try:
        # Charger le modèle et la configuration depuis Hugging Face
        model = MIDIModel.from_pretrained(model_name)
        tokenizer = model.tokenizer

        # Charger les poids LoRA si fournis
        if lora_path:
            model = model.load_merge_lora(lora_path)

        # Déplacer le modèle sur le bon dispositif
        model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
        model.eval()

        print("Modèle et tokenizer chargés avec succès.")
        return model, tokenizer
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        raise

def generate_midi(model: MIDIModel, tokenizer: MIDITokenizer, output_path: str, prompt: Optional[List[List[int]]] = None,
                 batch_size: int = 1, max_len: int = 512, temp: float = 1.0, top_p: float = 0.98, top_k: int = 20,
                 seed: Optional[int] = None, instruments: Optional[List[str]] = None, drum_kit: str = "None",
                 bpm: int = 0, time_sig: str = "auto", key_sig: str = "auto"):
    """
    Génère un fichier MIDI en utilisant le modèle chargé.

    Args:
        model (MIDIModel): Modèle MIDI chargé.
        tokenizer (MIDITokenizer): Tokenizer associé.
        output_path (str): Chemin de sortie pour le fichier MIDI généré.
        prompt (Optional[List[List[int]]], optional): Séquence de tokens de prompt. Defaults to None.
        batch_size (int, optional): Taille du batch. Defaults to 1.
        max_len (int, optional): Longueur maximale de la séquence générée. Defaults to 512.
        temp (float, optional): Température pour l'échantillonnage. Defaults to 1.0.
        top_p (float, optional): Valeur top-p pour la censure des probabilités. Defaults to 0.98.
        top_k (int, optional): Valeur top-k pour la censure des probabilités. Defaults to 20.
        seed (Optional[int], optional): Graine pour la reproductibilité. Defaults to None.
        instruments (Optional[List[str]], optional): Liste des instruments à utiliser. Defaults to None.
        drum_kit (str, optional): Kit de batterie à utiliser. Defaults to "None".
        bpm (int, optional): BPM (Beats Per Minute). Defaults to 0.
        time_sig (str, optional): Signature temporelle. Defaults to "auto".
        key_sig (str, optional): Signature de clé. Defaults to "auto".
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Préparer les paramètres du prompt
    prompt_params = {
        "instruments": instruments if instruments else [],
        "drum_kit": drum_kit,
        "bpm": bpm,
        "time_signature": time_sig,
        "key_signature": key_sig
    }

    # Intégrer les paramètres du prompt dans la séquence de tokens si nécessaire
    # Ceci dépend de la manière dont le modèle interprète les prompts
    # Vous devrez peut-être adapter cette partie selon votre modèle
    # Par exemple, ajouter des tokens spéciaux ou des balises pour les paramètres
    # Pour cet exemple, nous supposons que les paramètres sont déjà intégrés dans le prompt JSON

    midi_seq = model.generate(prompt=prompt, batch_size=batch_size, max_len=max_len, temp=temp,
                              top_p=top_p, top_k=top_k)

    # Detokeniser la séquence MIDI
    midi_score = tokenizer.detokenize(midi_seq[0])  # Prendre le premier batch

    # Convertir le score MIDI en fichier MIDI
    midi_file = MIDI.score2midi(midi_score)

    # Afficher le chemin de sortie pour le débogage
    print(f"Chemin de sortie : {output_path}")

    # Créer le répertoire de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder le fichier MIDI
    with open(output_path, 'wb') as f:
        f.write(midi_file)

    print(f"MIDI généré et sauvegardé à : {output_path}")

def generate_continuation_midi(
    model: MIDIModel,
    tokenizer: MIDITokenizer,
    output_path: str,
    input_midi_path: Optional[str] = None,
    batch_size: int = 1,
    max_len: int = 512,
    temp: float = 1.0,
    top_p: float = 0.98,
    top_k: int = 20,
    seed: Optional[int] = None,
    instruments: Optional[List[str]] = None,
    drum_kit: str = "None",
    bpm: int = 0,
    time_sig: str = "auto",
    key_sig: str = "auto"
):
    """
    Génère un fichier MIDI en continuant à partir d'un fichier MIDI existant.

    Args:
        model (MIDIModel): Modèle MIDI chargé.
        tokenizer (MIDITokenizer): Tokenizer associé.
        output_path (str): Chemin de sortie pour le fichier MIDI généré.
        input_midi_path (Optional[str], optional): Chemin vers le fichier MIDI d'entrée. Defaults to None.
        batch_size (int, optional): Taille du batch. Defaults to 1.
        max_len (int, optional): Nombre total d'événements MIDI à générer. Defaults to 512.
        temp (float, optional): Température pour l'échantillonnage. Defaults to 1.0.
        top_p (float, optional): Valeur top-p pour la censure des probabilités. Defaults to 0.98.
        top_k (int, optional): Valeur top-k pour la censure des probabilités. Defaults to 20.
        seed (Optional[int], optional): Graine pour la reproductibilité. Defaults to None.
        instruments (Optional[List[str]], optional): Liste des instruments à utiliser. Defaults to None.
        drum_kit (str, optional): Kit de batterie à utiliser. Defaults to "None".
        bpm (int, optional): BPM (Beats Per Minute). Defaults to 0.
        time_sig (str, optional): Signature temporelle. Defaults to "auto".
        key_sig (str, optional): Signature de clé. Defaults to "auto".
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Préparer le prompt à partir du fichier MIDI d'entrée
    if input_midi_path:
        with open(input_midi_path, 'rb') as f:
            midi_content = f.read()
        midi_score = MIDI.midi2score(midi_content)
        prompt_tokens = tokenizer.tokenize(
            midi_score,
            add_bos_eos=True,
            cc_eps=4,
            tempo_eps=4,
            remap_track_channel=True,
            add_default_instr=True,
            remove_empty_channels=False
        )
        prompt_length = len(prompt_tokens)
        prompt_tokens = prompt_tokens[:int(max_len / 2)]  # Utiliser une partie du MIDI comme prompt
        prompt_tokens = np.asarray([prompt_tokens] * batch_size, dtype=np.int64)
    else:
        prompt_tokens = None
        prompt_length = 0

    # Générer les nouveaux tokens (suite) sans inclure le prompt
    generated = model.generate(
        prompt=prompt_tokens,
        batch_size=batch_size,
        max_len=max_len,
        temp=temp,
        top_p=top_p,
        top_k=top_k
    )

    if input_midi_path:
        # Supposons que 'generated' inclut le prompt + continuation
        # Extraire uniquement les nouveaux tokens générés
        new_tokens = generated[:, prompt_length:]
    else:
        # Si aucun prompt, tout ce qui est généré est la suite
        new_tokens = generated

    # Vérifier si des nouveaux tokens ont été générés
    if new_tokens.size == 0:
        print("Aucun nouveau token généré.")
        return

    # Détokeniser uniquement les nouveaux tokens générés
    midi_score = tokenizer.detokenize(new_tokens[0])  # Détokeniser le premier batch

    # Convertir le score MIDI en fichier MIDI
    midi_file = MIDI.score2midi(midi_score)

    # Afficher le chemin de sortie pour le débogage
    print(f"Chemin de sortie : {output_path}")

    # Créer le répertoire de sortie si nécessaire
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder le fichier MIDI
    with open(output_path, 'wb') as f:
        f.write(midi_file)

    print(f"MIDI généré et sauvegardé à : {output_path}")

def create_prompt_from_midi(midi_file_path: str, tokenizer: MIDITokenizer, optimise_midi: bool = True) -> List[List[int]]:
    """
    Crée une séquence de tokens à partir d'un fichier MIDI.

    Args:
        midi_file_path (str): Chemin vers le fichier MIDI.
        tokenizer (MIDITokenizer): Tokenizer associé.
        optimise_midi (bool, optional): Optimisation des événements MIDI. Defaults to True.

    Returns:
        List[List[int]]: Séquence de tokens MIDI.
    """
    with open(midi_file_path, 'rb') as f:
        midi_content = f.read()
    midi_score = MIDI.midi2score(midi_content)
    midi_seq = tokenizer.tokenize(
        midi_score, add_bos_eos=True,
        cc_eps=4 if optimise_midi else 0,
        tempo_eps=4 if optimise_midi else 0,
        remap_track_channel=True,
        add_default_instr=True,
        remove_empty_channels=False
    )
    return midi_seq

def save_prompt(midi_seq: List[List[int]], output_json_path: str):
    """
    Sauvegarde la séquence de tokens MIDI dans un fichier JSON.

    Args:
        midi_seq (List[List[int]]): Séquence de tokens MIDI.
        output_json_path (str): Chemin de sortie pour le fichier JSON.
    """
    with open(output_json_path, 'w') as f:
        json.dump(midi_seq, f)
    print(f"Prompt sauvegardé à {output_json_path}")

def main():
    parser = argparse.ArgumentParser(description="Générateur de fichiers MIDI avec le modèle Hugging Face.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sous-commandes : generate, continue, create_prompt")

    # Sous-commande pour générer un nouveau MIDI
    parser_generate = subparsers.add_parser("generate", help="Générer un nouveau fichier MIDI.")
    parser_generate.add_argument("--model_name", type=str, required=True, help="Nom du modèle sur Hugging Face (e.g., skytnt/midi-model-tv2o-medium)")
    parser_generate.add_argument("--output", type=str, default="output.mid", help="Chemin de sortie pour le fichier MIDI généré.")
    parser_generate.add_argument("--prompt_json", type=str, default=None, help="Chemin vers un fichier JSON contenant la séquence de tokens de prompt.")
    parser_generate.add_argument("--batch_size", type=int, default=1, help="Taille du batch.")
    parser_generate.add_argument("--max_len", type=int, default=512, help="Longueur maximale de la séquence générée.")
    parser_generate.add_argument("--temp", type=float, default=1.0, help="Température pour l'échantillonnage.")
    parser_generate.add_argument("--top_p", type=float, default=0.98, help="Valeur top-p pour la censure des probabilités.")
    parser_generate.add_argument("--top_k", type=int, default=20, help="Valeur top-k pour la censure des probabilités.")
    parser_generate.add_argument("--seed", type=int, default=None, help="Graine pour la reproductibilité.")
    parser_generate.add_argument("--lora_path", type=str, default=None, help="Chemin vers les poids LoRA (optionnel).")

    # Options supplémentaires pour la génération
    parser_generate.add_argument("--instruments", type=str, nargs='*', help="Liste des instruments à utiliser.")
    parser_generate.add_argument("--drum_kit", type=str, default="None", choices=list(drum_kits2number.keys()), help="Kit de batterie à utiliser.")
    parser_generate.add_argument("--bpm", type=int, default=0, help="BPM (Beats Per Minute).")
    parser_generate.add_argument("--time_sig", type=str, default="auto", choices=["auto", "4/4", "2/4", "3/4", "6/4", "7/4",
                                                                                   "2/2", "3/2", "4/2", "3/8", "5/8",
                                                                                   "6/8", "7/8", "9/8", "12/8"],
                                 help="Signature temporelle.")
    parser_generate.add_argument("--key_sig", type=str, default="auto", choices=["auto"] + key_signatures, help="Signature de clé.")

    # Sous-commande pour continuer à partir d'un MIDI existant
    parser_continue = subparsers.add_parser("continue", help="Continuer la génération à partir d'un fichier MIDI existant.")
    parser_continue.add_argument("--model_name", type=str, required=True, help="Nom du modèle sur Hugging Face (e.g., skytnt/midi-model-tv2o-medium)")
    parser_continue.add_argument("--input_midi", type=str, required=True, help="Chemin vers le fichier MIDI d'entrée pour la continuation.")
    parser_continue.add_argument("--output", type=str, default="continued_output.mid", help="Chemin de sortie pour le fichier MIDI généré.")
    parser_continue.add_argument("--batch_size", type=int, default=1, help="Taille du batch.")
    parser_continue.add_argument("--max_len", type=int, default=512, help="Nombre total d'événements MIDI à générer.")
    parser_continue.add_argument("--temp", type=float, default=1.0, help="Température pour l'échantillonnage.")
    parser_continue.add_argument("--top_p", type=float, default=0.98, help="Valeur top-p pour la censure des probabilités.")
    parser_continue.add_argument("--top_k", type=int, default=20, help="Valeur top-k pour la censure des probabilités.")
    parser_continue.add_argument("--seed", type=int, default=None, help="Graine pour la reproductibilité.")
    parser_continue.add_argument("--lora_path", type=str, default=None, help="Chemin vers les poids LoRA (optionnel).")

    # Options supplémentaires pour la continuation
    parser_continue.add_argument("--instruments", type=str, nargs='*', help="Liste des instruments à utiliser.")
    parser_continue.add_argument("--drum_kit", type=str, default="None", choices=list(drum_kits2number.keys()), help="Kit de batterie à utiliser.")
    parser_continue.add_argument("--bpm", type=int, default=0, help="BPM (Beats Per Minute).")
    parser_continue.add_argument("--time_sig", type=str, default="auto", choices=["auto", "4/4", "2/4", "3/4", "6/4", "7/4",
                                                                                      "2/2", "3/2", "4/2", "3/8", "5/8",
                                                                                      "6/8", "7/8", "9/8", "12/8"],
                                  help="Signature temporelle.")
    parser_continue.add_argument("--key_sig", type=str, default="auto", choices=["auto"] + key_signatures, help="Signature de clé.")

    # Sous-commande pour créer un prompt à partir d'un fichier MIDI
    parser_create_prompt = subparsers.add_parser("create_prompt", help="Créer un fichier JSON de prompt à partir d'un fichier MIDI existant.")
    parser_create_prompt.add_argument("--model_name", type=str, required=True, help="Nom du modèle sur Hugging Face (e.g., skytnt/midi-model-tv2o-medium)")
    parser_create_prompt.add_argument("--midi_file", type=str, required=True, help="Chemin vers le fichier MIDI d'entrée.")
    parser_create_prompt.add_argument("--output_json", type=str, required=True, help="Chemin de sortie pour le fichier JSON de prompt.")
    parser_create_prompt.add_argument("--optimise_midi", action='store_true', help="Activer l'optimisation des événements MIDI.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Utilisation du dispositif : {device}")

    if args.command == "generate":
        # Charger le modèle et le tokenizer
        model, tokenizer = load_model_from_huggingface(args.model_name, lora_path=args.lora_path, device=device)

        # Charger le prompt depuis le fichier JSON si fourni
        prompt = None
        if args.prompt_json:
            with open(args.prompt_json, 'r') as f:
                prompt = json.load(f)
        
        # Générer le MIDI
        generate_midi(
            model, tokenizer, args.output, prompt=prompt,
            batch_size=args.batch_size, max_len=args.max_len,
            temp=args.temp, top_p=args.top_p, top_k=args.top_k,
            seed=args.seed, instruments=args.instruments,
            drum_kit=args.drum_kit, bpm=args.bpm,
            time_sig=args.time_sig, key_sig=args.key_sig
        )

    elif args.command == "continue":
        # Charger le modèle et le tokenizer
        model, tokenizer = load_model_from_huggingface(args.model_name, lora_path=args.lora_path, device=device)

        # Générer la continuation à partir du fichier MIDI existant
        generate_continuation_midi(
            model, tokenizer, args.output, input_midi_path=args.input_midi,
            batch_size=args.batch_size, max_len=args.max_len,
            temp=args.temp, top_p=args.top_p, top_k=args.top_k,
            seed=args.seed, instruments=args.instruments,
            drum_kit=args.drum_kit, bpm=args.bpm,
            time_sig=args.time_sig, key_sig=args.key_sig
        )

    elif args.command == "create_prompt":
        # Charger le modèle et le tokenizer
        model, tokenizer = load_model_from_huggingface(args.model_name, lora_path=None, device=device)

        # Créer le prompt à partir du fichier MIDI
        midi_seq = create_prompt_from_midi(args.midi_file, tokenizer, optimise_midi=args.optimise_midi)
        save_prompt(midi_seq, args.output_json)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
