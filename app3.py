import os
import sys
import json
import base64
import threading
import audioop
import queue
import numpy as np
import time
import asyncio
from scipy.signal import resample
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import httpx
import openai
import whisper  # Importamos Whisper para transcripción
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Pause
import uvicorn
import logging

# Configuración de logging
logging.basicConfig(level=logging.DEBUG)  # Cambiado a DEBUG para más detalles
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar credenciales
required_env_vars = [
    "ELEVENLABS_API_KEY",
    "VOICE_ID",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "NGROK_URL",
    "TWIML_BIN_URL",
]

missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(
        f"Las siguientes variables de entorno no están configuradas correctamente: {', '.join(missing_vars)}"
    )
    sys.exit(1)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_URL = os.getenv("NGROK_URL").rstrip('/')
TWIML_BIN_URL = os.getenv("TWIML_BIN_URL")

openai.api_key = OPENAI_API_KEY

app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Implementación de caché para respuestas y audios
response_cache = {}
audio_cache = {}

# Función para validar llamadas activas
def is_call_active(call_sid):
    try:
        call = twilio_client.calls(call_sid).fetch()
        logger.debug(f"Estado de la llamada: {call.status}")
        return call.status in ["in-progress", "ringing"]
    except Exception as e:
        logger.error(f"Verificando estado de la llamada: {e}")
        return False

# Función para procesar audio en tiempo real
def process_audio(audio_queue, stop_event, call_sid_container, loop):
    empty_transcription_count = 0
    try:
        # Cargamos el modelo de Whisper una sola vez
        model = whisper.load_model("tiny.en", device="cpu")
        # Inicializamos el buffer de audio
        audio_buffer = bytearray()
        packet_count = 0
        max_empty_transcriptions = 3

        while not stop_event.is_set():
            try:
                # Obtenemos audio de la cola
                audio_content = audio_queue.get(timeout=0.1)
                if audio_content:
                    # Guardamos el audio recibido para depuración
                    timestamp = int(time.time() * 1000)
                    with open(f"debug_audio_received_{timestamp}.raw", "wb") as f:
                        f.write(audio_content)

                    # Acumulamos los paquetes de audio
                    audio_buffer.extend(audio_content)
                    packet_count += 1

                    if packet_count >= 10:
                        # Tenemos suficientes paquetes para procesar
                        packet_count = 0

                        # Procesamos el audio acumulado
                        pcm_audio = audioop.ulaw2lin(bytes(audio_buffer), 2)

                        # Convertimos el audio PCM a un arreglo NumPy
                        audio_array = np.frombuffer(pcm_audio, dtype=np.int16)

                        # Si el audio tiene múltiples canales, convertirlo a mono
                        if audio_array.ndim > 1:
                            audio_array = audio_array.mean(axis=1)

                        # Log del tamaño del audio acumulado
                        logger.debug(f"Tamaño del audio acumulado: {len(audio_array)} muestras")

                        # Resampleamos el audio de 8 kHz a 16 kHz
                        original_sample_rate = 8000
                        target_sample_rate = 16000
                        number_of_samples = int(len(audio_array) * target_sample_rate / original_sample_rate)
                        audio_array_resampled = resample(audio_array, number_of_samples)

                        # Aseguramos que el arreglo es de tipo float32
                        audio_array_resampled = audio_array_resampled.astype(np.float32)

                        # Normalizamos el audio al rango entre -1 y 1
                        max_abs_val = np.max(np.abs(audio_array_resampled))
                        if max_abs_val > 0:
                            audio_array_resampled /= max_abs_val

                        # Logs detallados
                        logger.debug(f"Audio resampleado: {len(audio_array_resampled)} muestras.")
                        logger.debug(f"Máximo valor del audio: {np.max(audio_array_resampled)}")
                        logger.debug(f"Mínimo valor del audio: {np.min(audio_array_resampled)}")

                        # Detectar silencio en el audio
                        if np.max(np.abs(audio_array_resampled)) < 0.05:  # Umbral para detectar silencio
                            logger.warning("Audio silencioso detectado. Ignorando...")
                            # Limpiamos el buffer
                            audio_buffer = bytearray()
                            continue

                        # Procesamos el audio con Whisper
                        logger.info("Procesando audio con Whisper...")
                        result = model.transcribe(audio_array_resampled, language='en', fp16=False)

                        transcript = result.get('text', '').strip()
                        if not transcript:
                            empty_transcription_count += 1
                            logger.warning("Transcripción vacía. Intentando de nuevo...")
                            if empty_transcription_count >= max_empty_transcriptions:
                                logger.warning("Demasiadas transcripciones vacías consecutivas. Reproduciendo mensaje y continuando.")
                                # Reproducimos el mensaje y reiniciamos el contador
                                future = asyncio.run_coroutine_threadsafe(
                                    generate_and_play_audio("I couldn't hear you. Please try again.", call_sid_container["call_sid"]),
                                    loop
                                )
                                future.result()
                                empty_transcription_count = 0
                            # Limpiamos el buffer
                            audio_buffer = bytearray()
                            continue
                        else:
                            empty_transcription_count = 0

                        logger.info(f"Transcripción: {transcript}")
                        # Manejar la conversación
                        future = asyncio.run_coroutine_threadsafe(
                            handle_conversation(transcript, call_sid_container),
                            loop
                        )
                        future.result()

                        # Limpiamos el buffer después de procesar
                        audio_buffer = bytearray()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error procesando audio en tiempo real: {e}")
    except Exception as e:
        logger.error(f"Error en el hilo de procesamiento de audio: {e}")

# Función asincrónica para manejar la conversación
async def handle_conversation(transcript, call_sid_container):
    try:
        # Obtenemos respuesta de OpenAI
        openai_response = await get_openai_response_async(transcript)
        # Generamos y reproducimos el audio de la respuesta
        await generate_and_play_audio(openai_response, call_sid_container["call_sid"])
    except Exception as e:
        logger.error(f"Manejando conversación: {e}")

# Función asincrónica para obtener respuesta de OpenAI con streaming
async def get_openai_response_async(prompt):
    try:
        logger.info(f"Generando respuesta con OpenAI para el prompt: {prompt}")

        # Verificamos si la respuesta está en caché
        if prompt in response_cache:
            logger.info("Respuesta obtenida de la caché.")
            return response_cache[prompt]

        # Usamos streaming para recibir la respuesta progresivamente
        response_chunks = []
        async for chunk in await openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Mónica, an assistant for NetConnect Services."},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        ):
            content = chunk.choices[0].delta.get('content', '')
            response_chunks.append(content)

        answer = ''.join(response_chunks).strip()
        if not answer:
            logger.error("Respuesta de OpenAI vacía.")
            return "I'm sorry, I couldn't process your request."

        # Almacenamos la respuesta en la caché
        response_cache[prompt] = answer
        logger.info(f"Respuesta generada por OpenAI: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Generando respuesta con OpenAI: {e}")
        return "I'm sorry, I couldn't process your request."

# Función asincrónica para generar audio con Eleven Labs y reproducirlo
async def generate_and_play_audio(text, call_sid):
    try:
        if not text:
            logger.error("Texto para generar audio está vacío.")
            return

        # Verificamos si el audio ya está en caché
        if text in audio_cache:
            audio_path = audio_cache[text]
            logger.info("Audio obtenido de la caché.")
        else:
            # Dividimos el texto en fragmentos pequeños
            text_fragments = split_text(text, max_length=250)

            # Generamos audios para cada fragmento y los concatenamos
            audio_paths = []
            for fragment in text_fragments:
                audio_path = await generate_audio_fragment(fragment)
                if audio_path:
                    audio_paths.append(audio_path)
                else:
                    logger.error("No se pudo generar un fragmento de audio.")
                    return

            # Combinamos los fragmentos de audio en uno solo
            combined_audio_path = combine_audio_files(audio_paths)
            audio_cache[text] = combined_audio_path  # Almacenamos en caché
            audio_path = combined_audio_path

        # Reproducimos el audio al usuario
        play_audio_to_caller(call_sid, audio_path)
    except Exception as e:
        logger.error(f"Generando y reproduciendo audio: {e}")

# Función para dividir el texto en fragmentos
def split_text(text, max_length):
    words = text.split()
    fragments = []
    current_fragment = ''

    for word in words:
        if len(current_fragment) + len(word) + 1 <= max_length:
            current_fragment += ' ' + word if current_fragment else word
        else:
            fragments.append(current_fragment)
            current_fragment = word
    if current_fragment:
        fragments.append(current_fragment)
    return fragments

# Función asincrónica para generar audio de un fragmento de texto
async def generate_audio_fragment(text):
    try:
        logger.info(f"Generando audio con Eleven Labs para el texto: {text}")
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        data = {"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.75}}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=data, headers=headers)

            if response.status_code == 200:
                if not os.path.exists("static"):
                    os.makedirs("static")
                timestamp = int(time.time() * 1000)
                audio_filename = f"Response_{timestamp}.mp3"
                audio_path = os.path.join("static", audio_filename)
                with open(audio_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Audio generado y guardado: {audio_path}")
                return audio_path
            else:
                logger.error(f"Eleven Labs falló: {response.text}")
                return None
    except Exception as e:
        logger.error(f"Generando audio asincrónicamente: {e}")
        return None

# Función para combinar múltiples archivos de audio en uno solo
def combine_audio_files(audio_paths):
    from pydub import AudioSegment

    combined = AudioSegment.empty()
    for path in audio_paths:
        audio = AudioSegment.from_mp3(path)
        combined += audio

    combined_audio_path = os.path.join("static", f"Combined_{int(time.time() * 1000)}.mp3")
    combined.export(combined_audio_path, format="mp3")
    logger.info(f"Audio combinado guardado: {combined_audio_path}")

    # Limpiamos los fragmentos temporales
    for path in audio_paths:
        os.remove(path)

    return combined_audio_path

# Reproducir respuesta al usuario
def play_audio_to_caller(call_sid, audio_path):
    try:
        if not call_sid:
            logger.error("call_sid es None, no se puede reproducir el audio.")
            return

        if not is_call_active(call_sid):
            logger.error("La llamada no está activa. No se puede reproducir audio.")
            return

        # Verificar si el archivo de audio existe
        if not os.path.exists(audio_path):
            logger.error(f"El archivo de audio no existe: {audio_path}")
            return

        # Construir la URL del archivo de audio
        audio_url = f"https://{NGROK_URL}/static/{os.path.basename(audio_path)}"
        logger.info(f"URL del audio: {audio_url}")

        # Crear TwiML para reproducir el audio y mantener la llamada activa
        response = VoiceResponse()
        response.play(audio_url)
        response.pause(length=5)  # Pausa de 5 segundos para mantener la llamada activa
        twilio_client.calls(call_sid).update(twiml=str(response))
        logger.info("Audio reproducido correctamente al llamante.")
    except Exception as e:
        logger.error(f"Reproduciendo audio: {e}")

# Ruta para servir archivos estáticos
@app.get("/static/{filename}")
async def serve_static(filename: str):
    return FileResponse(path=f"static/{filename}")

# WebSocket para Twilio Streaming
@app.websocket("/media")
async def media_socket(websocket: WebSocket):
    logger.info("Conexión WebSocket establecida.")
    await websocket.accept()
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    call_sid_container = {"call_sid": None}

    # Enviamos mensajes keep-alive para mantener la conexión
    async def send_keep_alive():
        while not stop_event.is_set():
            await websocket.send_text('{"event": "keep-alive"}')
            await asyncio.sleep(10)

    # Iniciar tareas asincrónicas
    keep_alive_task = asyncio.create_task(send_keep_alive())
    loop = asyncio.get_running_loop()
    audio_thread = threading.Thread(
        target=process_audio, args=(audio_queue, stop_event, call_sid_container, loop), daemon=True
    )
    audio_thread.start()

    try:
        while not stop_event.is_set():
            try:
                message = await websocket.receive_text()
                if message is None:
                    logger.info("Conexión WebSocket cerrada por el cliente.")
                    break
                event_data = json.loads(message)
                event = event_data.get("event", "")
                logger.debug(f"Evento recibido: {event}")

                if event == "start":
                    call_sid_container["call_sid"] = event_data["start"]["callSid"]
                    logger.info(f"Call SID recibido: {call_sid_container['call_sid']}")
                elif event == "media":
                    payload = event_data["media"]["payload"]
                    audio_content = base64.b64decode(payload)
                    audio_queue.put(audio_content)
                    logger.debug(f"Audio recibido y agregado a la cola: {len(audio_content)} bytes")
                elif event == "stop":
                    logger.info("Evento 'stop' recibido. Cerrando conexión.")
                    stop_event.set()
                    break
                else:
                    logger.warning(f"Evento no manejado: {event}")
            except Exception as e:
                logger.error(f"En WebSocket: {e}")
                break
    except WebSocketDisconnect:
        logger.info("WebSocket desconectado.")
    finally:
        stop_event.set()
        await websocket.close()
        keep_alive_task.cancel()
        logger.info("WebSocket cerrado.")

if __name__ == "__main__":
    try:
        logger.info("Iniciando servidor...")
        uvicorn.run(app, host="0.0.0.0", port=5001)
    except Exception as e:
        logger.error(f"Error General: {str(e)}")