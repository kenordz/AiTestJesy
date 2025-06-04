# jesy_ai/tts.py
import os
import time
import logging
import httpx
from pydub import AudioSegment
from config import (
    ELEVENLABS_API_KEY,
    VOICE_ID,
    ELEVENLABS_STABILITY,
    ELEVENLABS_SIMILARITY_BOOST,
    MAX_CHAR_LIMIT,
    MAX_RETRIES,
    STATIC_AUDIO_PATH,
    ERROR_AUDIO_PATH
)
from utils import split_text

logger = logging.getLogger(__name__)

def generate_audio_sync(text):
    """
    Genera TTS con Eleven Labs de forma síncrona.
    """
    if not text or len(text.strip()) < 5:
        logger.error("Texto demasiado corto para TTS.")
        return None

    text = text[:5000]
    text_chunks = split_text(text, MAX_CHAR_LIMIT)
    audio_files = []

    for idx, chunk in enumerate(text_chunks):
        success = False
        attempts = 0
        while not success and attempts < MAX_RETRIES:
            attempts += 1
            try:
                url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY,
                }
                start_time = time.time()
                resp = httpx.post(url, json={"text": chunk, "voice_settings": {"stability": ELEVENLABS_STABILITY, "similarity_boost": ELEVENLABS_SIMILARITY_BOOST}}, headers=headers)
                end_time = time.time()
                logger.info(f"TTS chunk {idx+1}, intento {attempts}, duró {end_time - start_time:.2f}s")

                if resp.status_code == 200:
                    if not os.path.exists(STATIC_AUDIO_PATH):
                        os.makedirs(STATIC_AUDIO_PATH)
                    timestamp = int(time.time() * 1000)
                    temp_filename = f"Response_{timestamp}_{idx}.mp3"
                    temp_path = os.path.join(STATIC_AUDIO_PATH, temp_filename)
                    with open(temp_path, "wb") as f:
                        f.write(resp.content)
                    audio_files.append(temp_path)
                    success = True
                else:
                    logger.error(f"Error TTS chunk {idx+1}: {resp.text}")
            except Exception as exc:
                logger.error(f"Excepción TTS chunk {idx+1}: {exc}", exc_info=True)

        if not success:
            break

    if len(audio_files) != len(text_chunks):
        logger.error("No se generaron todos los fragmentos TTS.")
        if os.path.exists(ERROR_AUDIO_PATH):
            return ERROR_AUDIO_PATH
        return None

    final_name = f"Response_{int(time.time())}.mp3"
    final_path = os.path.join(STATIC_AUDIO_PATH, final_name)
    outcome = concatenate_audio_files_sync(audio_files, final_path)
    return outcome

def concatenate_audio_files_sync(audio_files, output_path):
    try:
        combined = AudioSegment.empty()
        for afile in audio_files:
            seg = AudioSegment.from_file(afile)
            combined += seg
        combined.export(output_path, format="mp3")
        logger.info(f"TTS concatenado -> {output_path}")
        for afile in audio_files:
            os.remove(afile)
        return output_path
    except Exception as e:
        logger.error(f"Error al concatenar TTS: {e}", exc_info=True)
        return None

async def generate_audio_async(text, audio_path=None):
    """
    Genera audio de forma asíncrona (útil para audios estáticos al arranque).
    """
    try:
        if not text or len(text.strip()) < 5:
            logger.error("Texto vacío/corto para audio estático.")
            return None

        text = text[:5000]
        text_chunks = split_text(text, MAX_CHAR_LIMIT)
        audio_files = []

        async with httpx.AsyncClient() as client:
            for idx, chunk in enumerate(text_chunks):
                success = False
                attempts = 0
                while not success and attempts < MAX_RETRIES:
                    attempts += 1
                    try:
                        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
                        headers = {
                            "Accept": "audio/mpeg",
                            "Content-Type": "application/json",
                            "xi-api-key": ELEVENLABS_API_KEY,
                        }
                        data = {"text": chunk, "voice_settings": {"stability": ELEVENLABS_STABILITY, "similarity_boost": ELEVENLABS_SIMILARITY_BOOST}}
                        start_t = time.time()
                        resp = await client.post(url, json=data, headers=headers)
                        end_t = time.time()
                        logger.info(f"Audio estático, chunk {idx+1}, intento {attempts}, duró {end_t - start_t:.2f}s")

                        if resp.status_code == 200:
                            if not os.path.exists(STATIC_AUDIO_PATH):
                                os.makedirs(STATIC_AUDIO_PATH)
                            timestamp = int(time.time() * 1000)
                            temp_filename = f"Response_{timestamp}_{idx}.mp3"
                            temp_path = os.path.join(STATIC_AUDIO_PATH, temp_filename)
                            with open(temp_path, "wb") as f:
                                f.write(resp.content)
                            audio_files.append(temp_path)
                            success = True
                        else:
                            logger.error(f"Error TTS estático chunk {idx+1}: {resp.text}")
                    except Exception as e:
                        logger.error(f"Excepción TTS estático chunk {idx+1}: {e}", exc_info=True)

                    if not success and attempts >= MAX_RETRIES:
                        logger.error("No se pudo generar uno de los fragmentos estáticos.")
                        break

        if len(audio_files) != len(text_chunks):
            logger.error("No se generaron todos los fragmentos audio estático.")
            if os.path.exists(ERROR_AUDIO_PATH):
                return ERROR_AUDIO_PATH
            return None

        if audio_path is None:
            audio_path = os.path.join(STATIC_AUDIO_PATH, f"Response_{int(time.time())}.mp3")

        final_path = concatenate_audio_files_sync(audio_files, audio_path)
        if final_path:
            logger.info(f"Audio estático final: {final_path}")
            return final_path
        else:
            logger.error("No se pudo concatenar el audio estático.")
            return None

    except Exception as e:
        logger.error(f"Error en generate_audio_async: {e}", exc_info=True)
        return None

async def generate_audio_file_static(text, audio_path):
    """
    Genera un archivo de audio estático (por ejemplo, welcome, wait, etc.).
    """
    logger.info(f"Generando audio estático para: {text}")
    final_path = await generate_audio_async(text, audio_path)
    if final_path:
        logger.info(f"Audio estático guardado: {final_path}")
    else:
        logger.error("No se pudo generar audio estático.")
    return final_path