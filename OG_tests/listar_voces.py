from google.cloud import texttospeech
import os

# Configura la ruta de las credenciales de Google Cloud


def listar_voces():
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices()

    for voice in response.voices:
        if "es-ES" in voice.language_codes:
            print(f"Nombre de voz: {voice.name}")
            print(f"Idioma(s): {voice.language_codes}")
            print(f"Tipo de voz: {voice.ssml_gender}")
            print(f"Frecuencia de muestra recomendada: {voice.natural_sample_rate_hertz}\n")

listar_voces()
