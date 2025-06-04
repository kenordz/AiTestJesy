import os
import requests
from dotenv import load_dotenv

# Cargar las credenciales del archivo .env
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = "tS2aLcsDt01lpGYODzAo"  # Reemplazado con el nuevo Voice ID

def generar_audio(texto, nombre_archivo):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": texto,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        with open(nombre_archivo, "wb") as archivo:
            archivo.write(response.content)
        print(f"Audio guardado como {nombre_archivo}")
    else:
        print(f"Error en la solicitud: {response.status_code} {response.json()}")

# Ejemplo de uso
generar_audio("Hello! Welcome to VOXAI Call Center, your trusted partner in customer support. My name is Monica, and I'm here to assist you. How can I help you today? Whether you have questions, need assistance, or want to know more about our services, I'm here to provide you with the answers you need. Thank you for choosing VOXAI!"
, "output_monica.wav")
