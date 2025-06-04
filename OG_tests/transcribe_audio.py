import os
from google.cloud import speech

# Configura las credenciales de Google Cloud


def transcribe_audio(audio_file_path):
    client = speech.SpeechClient()

    # Cargar el archivo de audio
    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()

    # Configuración de la solicitud de transcripción
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,  # Cambiado a 44100 Hz
        language_code="en-US"
    )

    # Solicitar la transcripción
    response = client.recognize(config=config, audio=audio)

    # Mostrar resultados
    if not response.results:
        print("No se encontraron resultados.")
    else:
        for result in response.results:
            print("Transcripción:", result.alternatives[0].transcript)

# Llamar a la función con la ruta correcta del archivo de audio
transcribe_audio("/Users/eugenio/Documents/Ai_Test_Audio.wav")
