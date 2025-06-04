import wave
import os

# Ruta al archivo de audio
audio_file_path = "/Users/eugenio/Documents/Ai_Test_Audio.wav"  # Ruta del archivo correcto

# Prueba de carga de audio usando wave
try:
    with wave.open(audio_file_path, 'rb') as audio:
        frames = audio.getnframes()
        rate = audio.getframerate()
        duration = frames / float(rate)
        print("Audio cargado exitosamente:", duration, "segundos")
except Exception as e:
    print("Error cargando el audio:", e)