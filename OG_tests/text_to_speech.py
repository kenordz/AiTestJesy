from google.cloud import texttospeech
import os

# Asegúrate de que la ruta de las credenciales esté configurada correctamente


def generar_audio(texto, archivo_salida):
    client = texttospeech.TextToSpeechClient()

    # Configuración de entrada de texto
    input_text = texttospeech.SynthesisInput(text=texto)

    # Selección de voz en inglés de Estados Unidos, con una voz Neural o Studio
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Studio-M"  # Cambia el nombre de la voz aquí si prefieres otra opción
    )

    # Configuración de audio de salida
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    # Generar la respuesta de síntesis de voz
    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    # Guardar el audio en el archivo de salida
    with open(archivo_salida, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio guardado como {archivo_salida}")

# Llamada a la función para generar el audio
generar_audio("Hello, How can I help you today?... Oh... its okey my name is Harry and im here to help!", "output_english.wav")
