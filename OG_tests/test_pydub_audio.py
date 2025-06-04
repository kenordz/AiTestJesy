from pydub import AudioSegment
import os

# Configura la ruta de ffmpeg si es necesario
AudioSegment.converter = "/usr/local/bin/ffmpeg"  # Asegúrate de que esta ruta es correcta

# Ruta al archivo de audio de prueba
audio_file_path = "/Users/eugenio/Documents/Prueba AI.wav"  # Ruta del archivo de prueba

# Prueba de carga de audio
try:
    audio = AudioSegment.from_file(audio_file_path, format="wav")  # Cambia "wav" si el formato es diferente
    print("Audio cargado exitosamente:", audio.duration_seconds, "segundos")
except Exception as e:
    print("Error cargando el audio:", e)