import os
import shutil

def collect_mid_files(source_dir, target_dir='data'):
    """
    Recherche récursivement tous les fichiers .mid dans source_dir et ses sous-répertoires,
    et les copie dans target_dir.
    
    Paramètres:
    - source_dir (str): Le chemin du répertoire source où chercher les fichiers .mid.
    - target_dir (str): Le chemin du répertoire cible où les fichiers seront copiés. Par défaut 'data'.
    """
    # Vérifier si le répertoire cible existe, sinon le créer
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Parcourir récursivement le répertoire source
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.mid'):
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


# Exemple d'utilisation
source_dir = 'User'
collect_mid_files(source_dir)

