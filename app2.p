import os
import sys
import json
import base64
import threading
import time
import asyncio
import re
from collections import deque  # Importar deque para manejar la cola de audio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocketState  # Importar WebSocketState para verificar el estado del WebSocket
from dotenv import load_dotenv
import httpx
import openai
from google.cloud import speech
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
import uvicorn
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)  # Mantener nivel INFO para reducir logs innecesarios
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar credenciales
required_env_vars = [
    "GOOGLE_APPLICATION_CREDENTIALS",
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

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_URL = os.getenv("NGROK_URL").rstrip('/')
TWIML_BIN_URL = os.getenv("TWIML_BIN_URL")

# Asegurarse de que NGROK_URL incluye el protocolo
if not NGROK_URL.startswith("http://") and not NGROK_URL.startswith("https://"):
    NGROK_URL = "https://" + NGROK_URL

openai.api_key = OPENAI_API_KEY

app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Implementación de caché para respuestas y audios
response_cache = {}  # Caché para respuestas de OpenAI
audio_cache = {}     # Caché para archivos de audio generados

# Caché para el estado de la llamada
call_status_cache = {}

# Cola para gestionar respuestas de OpenAI (para sincronización)
openai_response_queue = deque()  # Usando deque para la cola de OpenAI

# Variable para almacenar el último prompt procesado
last_generated_prompt = None

# Conjunto para almacenar las oraciones ya procesadas
processed_sentences = set()

# Función para validar llamadas activas con caché
def is_call_active(call_sid):
    try:
        current_time = time.time()
        cache_entry = call_status_cache.get(call_sid, {"timestamp": 0, "status": None})
        # Consultar estado cada 5 segundos para reducir llamadas a Twilio
        if current_time - cache_entry["timestamp"] > 5:
            call = twilio_client.calls(call_sid).fetch()
            cache_entry["status"] = call.status
            cache_entry["timestamp"] = current_time
            call_status_cache[call_sid] = cache_entry
            logger.debug(f"Estado de la llamada actualizado: {call.status}")
        else:
            logger.debug(f"Usando estado de llamada en caché: {cache_entry['status']}")
        return cache_entry["status"] in ["in-progress", "ringing"]
    except Exception as e:
        logger.error(f"Verificando estado de la llamada: {e}")
        return False

# Función para procesar audio en tiempo real
def process_audio(audio_queue, stop_event, call_sid_container, processing_complete_event):
    logger.debug("Iniciando process_audio")
    try:
        # Crear un cliente de Google Speech
        speech_client = speech.SpeechClient()
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            language_code="en-US",
            enable_automatic_punctuation=True,
            max_alternatives=1,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config, interim_results=True, single_utterance=False
        )

        # Generador de audio
        def audio_generator():
            buffer = bytearray()
            last_audio_time = time.time()
            silence_threshold = 0.5  # Reducir el tiempo de espera para enviar silencio
            min_chunk_size = 3200    # Tamaño mínimo del chunk para enviar
            while not stop_event.is_set():
                if len(audio_queue) > 0:
                    audio_content = audio_queue.popleft()
                    buffer.extend(audio_content)
                    last_audio_time = time.time()
                    if len(buffer) >= min_chunk_size:
                        yield speech.StreamingRecognizeRequest(audio_content=bytes(buffer))
                        logger.debug(f"Audio enviado al reconocedor: {len(buffer)} bytes")
                        buffer = bytearray()
                        time.sleep(0.1)  # Pausa para evitar sobresaturar gRPC
                else:
                    if (time.time() - last_audio_time) >= silence_threshold:
                        # Enviar silencio para mantener la conexión activa
                        silence = b"\xFF" * min_chunk_size
                        yield speech.StreamingRecognizeRequest(audio_content=silence)
                        logger.debug(f"Enviando silencio: {len(silence)} bytes")
                        last_audio_time = time.time()
                        time.sleep(0.1)  # Pausa para evitar sobresaturar gRPC

            # Enviar el resto del buffer al finalizar
            if buffer:
                yield speech.StreamingRecognizeRequest(audio_content=bytes(buffer))
                logger.debug(f"Audio final enviado al reconocedor: {len(buffer)} bytes")

        requests_generator = audio_generator()
        # Utilizar respuestas asíncronas
        responses = speech_client.streaming_recognize(streaming_config, requests_generator)

        # Manejar respuestas en un hilo separado
        threading.Thread(
            target=handle_responses,
            args=(responses, call_sid_container, stop_event, processing_complete_event),
            daemon=True
        ).start()
    except Exception as e:
        logger.error(f"Procesando audio en tiempo real: {e}")
        processing_complete_event.set()
    logger.debug("Finalizando process_audio")

# Función para manejar las respuestas del reconocedor de voz
def handle_responses(responses, call_sid_container, stop_event, processing_complete_event):
    logger.debug("Iniciando handle_responses")
    try:
        full_transcript = ""
        for response in responses:
            if not is_call_active(call_sid_container.get("call_sid")) or stop_event.is_set():
                logger.info("La llamada ha terminado o se ha solicitado detener el procesamiento.")
                break
            if not response.results:
                continue
            result = response.results[0]
            transcript = result.alternatives[0].transcript.strip()
            if result.is_final:
                full_transcript += " " + transcript
                logger.info(f"Transcripción final: {full_transcript.strip()}")
                if full_transcript.strip():
                    # Agregar el prompt a la cola de respuestas de OpenAI
                    openai_response_queue.append(full_transcript.strip())
                else:
                    logger.warning("No se obtuvo transcripción de voz.")
                # Reiniciar la transcripción para la siguiente frase
                full_transcript = ""
            else:
                # Evitar procesar transcripciones provisionales para optimizar recursos
                pass
    except Exception as e:
        logger.error(f"Manejo de respuestas: {e}")
    finally:
        processing_complete_event.set()
        logger.debug("processing_complete_event set")
    logger.debug("Finalizando handle_responses")

# Función para procesar la cola de respuestas de OpenAI
def process_openai_responses(call_sid_container, stop_event):
    logger.debug("Iniciando process_openai_responses")
    while not stop_event.is_set():
        try:
            prompt = openai_response_queue.popleft()
            get_openai_response(prompt, call_sid_container, stop_event)
        except IndexError:
            time.sleep(0.1)
            continue
    logger.debug("Finalizando process_openai_responses")

# Función para obtener respuesta de OpenAI
def get_openai_response(prompt, call_sid_container, stop_event):
    global last_generated_prompt
    logger.debug("Iniciando get_openai_response")
    try:
        if not is_call_active(call_sid_container.get("call_sid")) or stop_event.is_set():
            logger.info("La llamada ha terminado o se ha solicitado detener el procesamiento.")
            return

        if prompt == last_generated_prompt:
            logger.info("Prompt ya procesado anteriormente, evitando duplicados.")
            return
        last_generated_prompt = prompt

        logger.info(f"Generando respuesta con OpenAI para el prompt: {prompt}")
        start_time = time.time()

        # Verificar si la respuesta está en caché
        cached_response = response_cache.get(prompt.lower())
        if cached_response:
            logger.info("Respuesta obtenida de la caché.")
            generate_and_play_audio(cached_response, call_sid_container, stop_event)
            return

        # Usar la API de OpenAI sin streaming para simplificar y optimizar
        try:
            openai_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an assistant."},
                    {"role": "user", "content": prompt},
                ],
                timeout=15  # Añadir timeout para evitar esperas indefinidas
            )
            full_response = openai_response.choices[0].message.content.strip()
            logger.debug(f"Respuesta de OpenAI: {full_response}")
        except Exception as e:
            logger.error(f"Error en la llamada a OpenAI: {e}")
            generate_and_play_audio(
                "I'm sorry, there was an error processing your request.",
                call_sid_container,
                stop_event
            )
            return

        end_time = time.time()
        logger.info(f"Tiempo total en obtener respuesta de OpenAI: {end_time - start_time:.2f} segundos")

        # Almacenar la respuesta en la caché
        response_cache[prompt.lower()] = full_response

        # Generar y reproducir audio para la respuesta
        if full_response:
            generate_and_play_audio(full_response, call_sid_container, stop_event)
        else:
            logger.error("Respuesta de OpenAI vacía.")
            generate_and_play_audio(
                "I'm sorry, I couldn't process your request.",
                call_sid_container,
                stop_event
            )
    except Exception as e:
        logger.error(f"Generando respuesta con OpenAI: {e}")
        generate_and_play_audio(
            "I'm sorry, there was an error processing your request.",
            call_sid_container,
            stop_event
        )
    logger.debug("Finalizando get_openai_response")

# Función para generar audio y reproducirlo
def generate_and_play_audio(text, call_sid_container, stop_event):
    global processed_sentences
    logger.debug("Iniciando generate_and_play_audio")
    try:
        sentences = split_text_into_sentences(text)
        for sentence in reversed(sentences):
            if not is_call_active(call_sid_container.get("call_sid")) or stop_event.is_set():
                logger.info("La llamada ha terminado o se ha solicitado detener el procesamiento.")
                break

            if sentence.lower() in processed_sentences:
                logger.debug(f"Saltando oración ya procesada: {sentence}")
                continue

            processed_sentences.add(sentence.lower())
            threading.Thread(
                target=generate_audio_and_play,
                args=(sentence, call_sid_container, stop_event),
                daemon=True
            ).start()
    except Exception as e:
        logger.error(f"Generando y reproduciendo audio: {e}")
    logger.debug("Finalizando generate_and_play_audio")

def generate_audio_and_play(sentence, call_sid_container, stop_event):
    try:
        # Verificar si el audio ya está en caché y si el archivo existe
        cached_audio = audio_cache.get(sentence.lower())
        if cached_audio and os.path.exists(cached_audio):
            audio_path = cached_audio
            logger.info(f"Audio obtenido de la caché para el fragmento: {sentence}")
        else:
            audio_path = generate_audio(sentence)
            if audio_path and os.path.exists(audio_path):
                audio_cache[sentence.lower()] = audio_path
            else:
                logger.error("No se pudo generar o encontrar el audio para el fragmento.")
                return

        if audio_path:
            play_audio_to_caller(call_sid_container["call_sid"], audio_path)
        else:
            logger.error("No se pudo obtener la ruta del audio.")
    except Exception as e:
        logger.error(f"Generando y reproduciendo audio: {e}")

# Función para dividir el texto en oraciones o fragmentos más pequeños
def split_text_into_sentences(text, max_length=200):
    sentences = re.split('(?<=[.!?]) +', text)
    short_sentences = []
    for sentence in sentences:
        if len(sentence) > max_length:
            # Dividir en fragmentos más pequeños si la oración es muy larga
            short_sentences.extend([sentence[i:i+max_length] for i in range(0, len(sentence), max_length)])
        else:
            short_sentences.append(sentence)
    return short_sentences

# Función para generar audio con Eleven Labs
def generate_audio(text):
    logger.debug("Iniciando generate_audio")
    try:
        if not text:
            logger.error("Texto para generar audio está vacío.")
            return None
        logger.info(f"Generando audio con Eleven Labs para el texto: {text}")
        start_time = time.time()
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        data = {"text": text, "voice_settings": {"stability": 0.7, "similarity_boost": 0.75}}

        with httpx.Client(timeout=60) as client:
            response = client.post(url, json=data, headers=headers)

        end_time = time.time()
        logger.info(f"Tiempo en generar audio con Eleven Labs: {end_time - start_time:.2f} segundos")

        if response.status_code == 200 and response.content:
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
        logger.error(f"Generando audio: {e}")
        return None
    logger.debug("Finalizando generate_audio")

# Reproducir respuesta al usuario
def play_audio_to_caller(call_sid, audio_path):
    logger.debug("Iniciando play_audio_to_caller")
    try:
        if not call_sid:
            logger.error("call_sid es None, no se puede reproducir el audio.")
            return

        # Verificar y registrar el estado de la llamada
        if not is_call_active(call_sid):
            logger.error("La llamada no está activa, deteniendo reproducción.")
            return

        # Verificar si el archivo de audio existe
        if not os.path.exists(audio_path):
            logger.error(f"El archivo de audio no existe: {audio_path}")
            return

        # Construir la URL del archivo de audio
        audio_url = f"{NGROK_URL}/static/{os.path.basename(audio_path)}"
        logger.info(f"URL del audio: {audio_url}")

        # Validar que el archivo es accesible públicamente
        try:
            response = httpx.get(audio_url, timeout=10)
            if response.status_code != 200:
                logger.error(f"El archivo de audio no es accesible públicamente: {audio_url}")
                return
            else:
                logger.debug(f"El archivo de audio es accesible: {audio_url}")
        except Exception as e:
            logger.error(f"Error al acceder al archivo de audio: {e}")
            return

        # Crear TwiML para reproducir el audio
        response = VoiceResponse()
        response.play(audio_url)
        logger.debug(f"TwiML generado: {str(response)}")

        # Actualizar la llamada con TwiML
        call = twilio_client.calls(call_sid).update(twiml=str(response))
        logger.info(f"Audio reproducido correctamente al llamante. Estado de la llamada: {call.status}")
    except Exception as e:
        logger.error(f"Reproduciendo audio: {e}")
    logger.debug("Finalizando play_audio_to_caller")

# Ruta para servir archivos estáticos
@app.get("/static/{filename}")
async def serve_static(filename: str):
    file_path = f"static/{filename}"
    if os.path.exists(file_path):
        logger.debug(f"Sirviendo archivo estático: {file_path}")
        return FileResponse(path=file_path)
    else:
        logger.error(f"Archivo estático no encontrado: {file_path}")
        return {"error": "Archivo no encontrado"}

# WebSocket para Twilio Streaming
@app.websocket("/media")
async def media_socket(websocket: WebSocket):
    logger.info("Conexión WebSocket establecida.")
    await websocket.accept()
    audio_queue = deque(maxlen=100)  # Usando deque con tamaño máximo para la cola de audio
    stop_event = threading.Event()
    processing_complete_event = threading.Event()
    call_sid_container = {"call_sid": None}

    # Iniciar el hilo de procesamiento de audio
    threading.Thread(
        target=process_audio,
        args=(audio_queue, stop_event, call_sid_container, processing_complete_event),
        daemon=True
    ).start()

    # Iniciar el hilo para procesar respuestas de OpenAI
    threading.Thread(
        target=process_openai_responses,
        args=(call_sid_container, stop_event),
        daemon=True
    ).start()

    # Iniciar una tarea para enviar mensajes keep-alive
    async def keep_alive():
        while not stop_event.is_set():
            try:
                await websocket.send_text(json.dumps({"event": "keep-alive"}))
                logger.debug("Mensaje keep-alive enviado.")
                await asyncio.sleep(5)
            except Exception as e:
                logger.error(f"Error enviando mensaje keep-alive: {e}")
                break

    asyncio.create_task(keep_alive())

    try:
        while not stop_event.is_set():
            try:
                message = await websocket.receive_text()
                if message is None:
                    logger.info("Conexión WebSocket cerrada por el cliente.")
                    break
                logger.debug("Procesando evento WebSocket.")
                logger.info(f"Estado actual de la llamada: {is_call_active(call_sid_container.get('call_sid'))}")
                event_data = json.loads(message)
                event = event_data.get("event", "")
                if event == "connected":
                    logger.info("Evento 'connected' recibido.")
                    continue
                elif event == "start":
                    call_sid_container["call_sid"] = event_data["start"]["callSid"]
                    logger.info(f"Call SID recibido: {call_sid_container['call_sid']}")
                elif event == "media":
                    if not call_sid_container.get("call_sid"):
                        logger.error("Call SID no configurado. Ignorando evento 'media'.")
                        continue
                    # Manejar eventos 'media' de manera eficiente
                    payload = event_data["media"]["payload"]
                    audio_content = base64.b64decode(payload)
                    audio_queue.append(audio_content)
                elif event == "stop":
                    logger.info("Evento 'stop' recibido. Esperando a que el procesamiento finalice.")
                    stop_event.set()
                    # Esperar a que el procesamiento se complete
                    processing_complete_event.wait(timeout=10)  # Incrementar el tiempo de espera
                    await asyncio.sleep(2)  # Retraso adicional para evitar conflictos
                    break
                else:
                    logger.warning(f"Evento no manejado: {event}")
            except Exception as e:
                logger.error(f"En WebSocket: {e}")
                break
    except WebSocketDisconnect:
        logger.info("WebSocket desconectado.")
    finally:
        # Esperar a que todo el procesamiento se complete antes de cerrar el WebSocket
        processing_complete_event.wait()
        if not websocket.client_state == WebSocketState.DISCONNECTED:
            await websocket.close()
        logger.info("WebSocket cerrado.")

if __name__ == "__main__":
    try:
        logger.info("Iniciando servidor...")
        uvicorn.run(app, host="0.0.0.0", port=5001)
    except Exception as e:
        logger.error(f"Error General: {str(e)}")