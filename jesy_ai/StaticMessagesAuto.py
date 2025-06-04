import os
import time
import httpx
import random
import logging
from dotenv import load_dotenv
from pydub import AudioSegment

# ---------------------------------------------------------------------
# Cargar las variables de entorno (si tienes un .env con ELEVENLABS_API_KEY, VOICE_ID_ES, etc.)
# ---------------------------------------------------------------------
load_dotenv()

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID_ES = os.getenv("VOICE_ID_ES")

# Opcional: si no lo tienes en .env, puedes ponerlo “hardcodeado”:
# ELEVENLABS_API_KEY = "TU_API_KEY"
# VOICE_ID_ES = "TU_VOICE_ID"

# Parámetros recomendados (ajusta a tu gusto)
ELEVENLABS_STABILITY = 0.1
ELEVENLABS_SIMILARITY_BOOST = 0.9
MODEL_ID = "eleven_multilingual_v1"  # Para TTS en español
MAX_CHAR_LIMIT = 300
MAX_RETRIES = 3

# Carpeta donde vas a guardar los MP3
STATIC_AUDIO_PATH = "static"

# ---------------------------------------------------------------------
# Diccionario con tus mensajes en español
# ---------------------------------------------------------------------
STATIC_MESSAGES = {
    "bienvenido": "¡Gracias por marcar a SuperSalads! ¿En qué puedo ayudarte hoy?",
    "espera": [
        "Un momento, por favor.",
        "Dame un segundo, por favor.",
        "Enseguida te ayudo.",
        "Permíteme revisar, un momento."
    ],
    "errores": "Estamos experimentando dificultades técnicas. Por favor, intenta más tarde.",
}

# Configurar logging básico (para ver en consola)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Pequeña función para trocear el texto en pedazos <= MAX_CHAR_LIMIT
# ---------------------------------------------------------------------
def split_text(text: str, max_length: int):
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += (" " + word) if current_chunk else word
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ---------------------------------------------------------------------
# Generar TTS con ElevenLabs y guardar en un archivo .mp3
# ---------------------------------------------------------------------
def generate_audio_elevenlabs(text: str, output_path: str) -> bool:
    """
    Genera TTS con ElevenLabs y guarda el resultado en output_path.
    Si el texto es muy largo, lo trocea y concatena.
    """
    text = text.strip()
    if len(text) < 5:
        logger.warning(f"Texto demasiado corto: '{text}'")
        return False

    # Trocear si es más largo que 300 caracteres
    text = text[:5000]  # Límite para no exceder
    text_chunks = split_text(text, MAX_CHAR_LIMIT)
    audio_files = []

    # Asegurarnos de que exista la carpeta 'static'
    if not os.path.exists(STATIC_AUDIO_PATH):
        os.makedirs(STATIC_AUDIO_PATH, exist_ok=True)

    # Iterar cada chunk y enviarlo a ElevenLabs
    for idx, chunk in enumerate(text_chunks):
        success = False
        attempts = 0
        while attempts < MAX_RETRIES and not success:
            attempts += 1
            try:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID_ES}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY,
                }
                data = {
                    "text": chunk,
                    "voice_settings": {
                        "stability": ELEVENLABS_STABILITY,
                        "similarity_boost": ELEVENLABS_SIMILARITY_BOOST,
                    },
                    "model_id": MODEL_ID
                }
                t0 = time.time()
                resp = httpx.post(url, json=data, headers=headers)
                t1 = time.time()

                logger.info(f"[TTS] Chunk {idx+1}/{len(text_chunks)} - intento {attempts} - {t1 - t0:.2f}s")

                if resp.status_code == 200:
                    temp_filename = f"temp_{int(time.time()*1000)}_{random.randint(1000,9999)}.mp3"
                    temp_path = os.path.join(STATIC_AUDIO_PATH, temp_filename)
                    with open(temp_path, "wb") as f:
                        f.write(resp.content)
                    audio_files.append(temp_path)
                    success = True
                else:
                    logger.error(f"Error TTS chunk {idx+1}: {resp.text}")
            except Exception as exc:
                logger.error(f"Excepción TTS chunk {idx+1}: {exc}")

        if not success:
            logger.error(f"No se pudo generar audio tras {MAX_RETRIES} intentos para chunk {idx+1}")
            return False

    # Si todo bien, concatenar los fragmentos en output_path final
    if len(audio_files) == len(text_chunks):
        combined = AudioSegment.empty()
        for temp_file in audio_files:
            seg = AudioSegment.from_file(temp_file)
            combined += seg

        combined.export(output_path, format="mp3")
        logger.info(f"Archivo final generado: {output_path}")

        # Borrar archivos temporales
        for tf in audio_files:
            os.remove(tf)

        return True
    else:
        logger.error("No se generaron todos los fragmentos, no se hará concatenación final.")
        return False

# ---------------------------------------------------------------------
# Función principal para generar todos los audios de STATIC_MESSAGES
# ---------------------------------------------------------------------
def main():
    # Crear la carpeta static si no existe
    os.makedirs(STATIC_AUDIO_PATH, exist_ok=True)

    for key, value in STATIC_MESSAGES.items():
        # Caso 1: value es string
        if isinstance(value, str):
            output_mp3 = os.path.join(STATIC_AUDIO_PATH, f"{key}.mp3")
            logger.info(f"Generando audio para '{key}' => {output_mp3}")
            generate_audio_elevenlabs(value, output_mp3)

        # Caso 2: value es lista (por ejemplo "espera")
        elif isinstance(value, list):
            for idx, phrase in enumerate(value):
                output_mp3 = os.path.join(STATIC_AUDIO_PATH, f"{key}_{idx}.mp3")
                logger.info(f"Generando audio para '{key}_{idx}' => {output_mp3}")
                generate_audio_elevenlabs(phrase, output_mp3)

        else:
            logger.warning(f"Tipo de valor inesperado en STATIC_MESSAGES[{key}]: {type(value)}")

    logger.info("¡Listo! Se generaron los audios estáticos en la carpeta 'static'.")

if __name__ == "__main__":
    main()
    