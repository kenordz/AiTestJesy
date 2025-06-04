import os
from elevenlabs import text_to_speech
from dotenv import load_dotenv

load_dotenv()  # Cargar variables del .env

api_key = os.getenv("ELEVENLABS_API_KEY")  # Asegurarse de tener la clave en .env

def generar_audio(texto, voice="Monica"):
    try:
        audio = text_to_speech(api_key=api_key, text=texto, voice=voice)
        audio_path = "output_audio.mp3"
        with open(audio_path, "wb") as file:
            file.write(audio)
        print("Audio generado correctamente en", audio_path)
    except Exception as e:
        print("Error al generar el audio:", e)

mensaje_inicial = "Hello! Welcome to Netconnect Internet Services. Thank you for calling! My name is Monica, how can I assist you today?"

# Genera el audio
generar_audio(mensaje_inicial)