# Utiliser une image de base officielle Python
FROM python:3.10-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Installer git, TiMidity++ et ffmpeg
RUN apt-get update && \
    apt-get install -y git timidity ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y \
        libportaudio2 \
        libfluidsynth-dev \
        libportaudiocpp0 &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Copier tout le code de l'application dans le conteneur
COPY . .

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Commande pour lancer l'application
ENTRYPOINT ["streamlit", "run", "streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
