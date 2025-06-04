
import os
from google.cloud import speech

# Configura las credenciales


def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    # Cargar el archivo de audio
    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()

    # Configuración de la solicitud
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )

    # Solicitar transcripción
    response = client.recognize(config=config, audio=audio)

    # Mostrar resultados
    if not response.results:
        print("No se encontraron resultados.")
    else:
        for result in response.results:
            print("Transcription: {}".format(result.alternatives[0].transcript))

# Llamada a la función
transcribe_audio("/Users/eugenio/Documents/Prueba AI.wav")