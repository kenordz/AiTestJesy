from google.cloud import speech
import os

# Asegura que la ruta de las credenciales esté configurada correctamente


def transcribe_audio(audio_file_path):
    # Verifica si el archivo existe
    if not os.path.isfile(audio_file_path):
        print(f"El archivo de audio no se encontró en la ruta especificada: {audio_file_path}")
        return

    client = speech.SpeechClient()

    # Cargar el archivo de audio
    try:
        with open(audio_file_path, "rb") as audio_file:
            audio_content = audio_file.read()
    except Exception as e:
        print(f"Error al cargar el archivo de audio: {e}")
        return

    # Configuración de la solicitud
    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,  # Actualizado para coincidir con el archivo
        language_code="es-MX",
    )

    # Solicitar transcripción
    try:
        response = client.recognize(config=config, audio=audio)
        print("Solicitud de transcripción enviada correctamente.")
    except Exception as e:
        print(f"Error en la solicitud de transcripción: {e}")
        return

    # Mostrar resultados
    if response.results:
        for result in response.results:
            print("Transcription: {}".format(result.alternatives[0].transcript))
    else:
        print("No se obtuvo ninguna transcripción del audio.")

# Llamada a la función con la ruta correcta al archivo de audio en Downloads
audio_path = "/Users/eugenio/Downloads/PruebaAI.wav"
print(f"Ejecutando script con ruta de archivo: {audio_path}")
transcribe_audio(audio_path)
