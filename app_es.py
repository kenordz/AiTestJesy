import os
import sys
import json
import base64
import threading
import queue
import time
import asyncio
import random  # Importamos random para la selección aleatoria de frases
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import httpx
import openai
from google.cloud import speech
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse
import uvicorn
import logging
from pydub import AudioSegment  # Importamos pydub para manejar archivos de audio

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configurar credenciales
required_env_vars = [
    "GOOGLE_APPLICATION_CREDENTIALS",
    "ELEVENLABS_API_KEY",
    "VOICE_IDES",  # Se mantiene VOICE_IDES como en el original
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
VOICE_IDES = os.getenv("VOICE_IDES")  # VOICE_IDES se mantiene, debe ser un voice_id válido en Eleven Labs
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_URL = os.getenv("NGROK_URL").rstrip('/')
TWIML_BIN_URL = os.getenv("TWIML_BIN_URL")

openai.api_key = OPENAI_API_KEY

app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Parámetros para Eleven Labs (Ajustes mínimos para fluidez)
ELEVENLABS_STABILITY = 0.15  # Corregido: Aumentamos ligeramente la estabilidad
ELEVENLABS_SIMILARITY_BOOST = 0.95  # Corregido: Mayor similitud para voz más consistente

MAX_CHAR_LIMIT = 300
MAX_RETRIES = 3

STATIC_AUDIO_PATH = "static"
ERROR_AUDIO_PATH = os.path.join(STATIC_AUDIO_PATH, "errores.mp3")  # Corregido: errores.mp3 en español

# Mensajes estáticos en español
# Corregido: Todos los mensajes en español, con nombres de llaves en español
STATIC_MESSAGES = {
    "bienvenido": "Gracias por contactar con NetConnect Servicios, ¿en qué puedo ayudarle hoy?",
    "espera": [
        "Un momento, por favor.",
        "Ok, deme un segundo, por favor.",
        "Espere un momento, por favor.",
        "Un segundo, por favor.",
        "De acuerdo, permítame revisarlo, un segundo."
    ],
    "errores": "Estamos experimentando dificultades técnicas. Por favor, inténtelo más tarde.",
}

async def generate_static_audio():
    """
    Genera los archivos de audio para los mensajes estáticos si no existen.
    """
    if not os.path.exists(STATIC_AUDIO_PATH):
        os.makedirs(STATIC_AUDIO_PATH)

    for key, text in STATIC_MESSAGES.items():
        if key == 'espera':
            for idx, phrase in enumerate(text):
                audio_filename = f"{key}_{idx}.mp3"
                audio_path = os.path.join(STATIC_AUDIO_PATH, audio_filename)
                if not os.path.exists(audio_path):
                    logger.info(f"Generando audio estático para 'espera': {phrase}")
                    await generate_audio_file(phrase, audio_path)
                    logger.info(f"Audio estático generado: {audio_path}")
                else:
                    logger.info(f"Audio estático ya existe: {audio_path}")
        else:
            audio_filename = f"{key}.mp3"
            audio_path = os.path.join(STATIC_AUDIO_PATH, audio_filename)
            if not os.path.exists(audio_path):
                logger.info(f"Generando audio estático para '{key}': {text}")
                await generate_audio_file(text, audio_path)
                logger.info(f"Audio estático generado: {audio_path}")
            else:
                logger.info(f"Audio estático ya existe: {audio_path}")

async def generate_audio_file(text, audio_path):
    """
    Genera un archivo de audio para un texto dado (mensaje estático).
    """
    logger.info(f"Generando audio para mensaje estático: {text}")
    try:
        audio_generated_path = await generate_audio_async(text, audio_path)
        if audio_generated_path:
            logger.info(f"Audio estático generado y guardado en: {audio_generated_path}")
        else:
            logger.error("Error al generar el audio estático.")
    except Exception as e:
        logger.error(f"Error al generar audio estático: {e}", exc_info=True)

@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio de la aplicación: genera los audios estáticos si no existen.
    """
    await generate_static_audio()

def is_call_active(call_sid):
    """
    Verifica si la llamada sigue activa.
    """
    try:
        call = twilio_client.calls(call_sid).fetch()
        logger.debug(f"Estado de la llamada: {call.status}")
        return call.status in ["in-progress", "ringing"]
    except Exception as e:
        logger.error(f"Verificando estado de la llamada: {e}", exc_info=True)
        return False

def process_audio(audio_queue, stop_event, call_context):
    """
    Procesa el audio entrante usando Google STT (es-ES).
    """
    try:
        speech_client = speech.SpeechClient()
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            language_code="es-ES",  # en español
            enable_automatic_punctuation=True,
            max_alternatives=1,
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False,
            single_utterance=False,
        )

        def audio_generator():
            buffer = bytearray()
            last_audio_time = time.time()
            while not stop_event.is_set():
                try:
                    audio_content = audio_queue.get(timeout=0.005)
                    buffer.extend(audio_content)
                    last_audio_time = time.time()
                    if len(buffer) >= 320:
                        yield speech.StreamingRecognizeRequest(audio_content=bytes(buffer))
                        logger.debug(f"Audio enviado al reconocedor: {len(buffer)} bytes")
                        buffer = bytearray()
                except queue.Empty:
                    if (time.time() - last_audio_time) >= 0.03:
                        logger.debug("Enviando silencio para mantener la conexión.")
                        silence = b"\xFF" * 320
                        yield speech.StreamingRecognizeRequest(audio_content=silence)
                        last_audio_time = time.time()

        requests_generator = audio_generator()
        responses = speech_client.streaming_recognize(streaming_config, requests_generator)
        asyncio.run(handle_responses_async(responses, call_context, stop_event))
    except Exception as e:
        logger.error(f"Procesando audio en tiempo real: {e}", exc_info=True)

async def handle_responses_async(responses, call_context, stop_event):
    """
    Maneja las transcripciones del STT y genera respuestas OpenAI + audio.
    """
    try:
        wait_message_sent = False
        for response in responses:
            if not response.results:
                continue
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript.strip()
                    if not transcript or len(transcript) < 3:
                        logger.info("Transcripción vacía o muy corta. Ignorando.")
                        continue
                    logger.info(f"Transcripción final: {transcript}")

                    await asyncio.sleep(2)

                    if not wait_message_sent:
                        espera_phrases = STATIC_MESSAGES['espera']
                        selected_phrase = random.choice(espera_phrases)  # Corregido: se usa espera_phrases
                        idx = espera_phrases.index(selected_phrase)
                        wait_audio_filename = f"espera_{idx}.mp3"
                        wait_audio_path = os.path.join(STATIC_AUDIO_PATH, wait_audio_filename)
                        if os.path.exists(wait_audio_path):
                            play_audio_to_caller(call_context["call_sid"], wait_audio_path)
                            wait_message_sent = True
                        else:
                            logger.error(f"Archivo de audio estático 'espera' no existe: {wait_audio_path}")

                    call_context["conversation_history"].append({"role": "user", "content": transcript})
                    call_context["conversation_history"] = await trim_conversation_history(
                        call_context["conversation_history"]
                    )

                    openai_task = asyncio.create_task(get_openai_response_async(call_context["conversation_history"]))
                    openai_response = await openai_task

                    call_context["conversation_history"].append({"role": "assistant", "content": openai_response})

                    audio_task = asyncio.create_task(generate_audio_async(openai_response))
                    response_audio_path = await audio_task

                    if response_audio_path:
                        play_audio_to_caller(call_context["call_sid"], response_audio_path)
                    else:
                        logger.error("No se pudo generar audio de la respuesta.")
                        if os.path.exists(ERROR_AUDIO_PATH):
                            play_audio_to_caller(call_context["call_sid"], ERROR_AUDIO_PATH)
                        else:
                            logger.error("El archivo de audio de error predeterminado no existe.")

                    wait_message_sent = False

                    if stop_event.is_set():
                        break
    except Exception as e:
        logger.error(f"Manejo de respuestas asincrónicas: {e}", exc_info=True)

async def trim_conversation_history(conversation_history, max_tokens=4000):
    """
    Recorta el historial si excede el límite de tokens.
    """
    total_tokens = sum(len(m["content"].split()) for m in conversation_history)
    while total_tokens > max_tokens:
        if len(conversation_history) > 1:
            removed_message = conversation_history.pop(1)
            logger.debug(f"Eliminando mensaje antiguo: {removed_message}")
            total_tokens = sum(len(m["content"].split()) for m in conversation_history)
        else:
            break
    return conversation_history

async def get_openai_response_async(conversation_history):
    """
    Envía el historial a OpenAI y obtiene una respuesta en español.
    Implementamos un reintento simple en caso de error.
    """
    attempts = 0
    while attempts < 2:  # Corregido: Agregamos reintento simple si falla la solicitud a OpenAI
        try:
            logger.info("Generando respuesta con OpenAI para el historial.")
            loop = asyncio.get_event_loop()
            start_time = time.time()
            response = await loop.run_in_executor(
                None,
                lambda: openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=conversation_history,
                ),
            )
            end_time = time.time()
            logger.info(f"Tiempo en obtener respuesta de OpenAI: {end_time - start_time:.2f} segundos")
            answer = response["choices"][0]["message"]["content"].strip()
            if not answer:
                logger.error("Respuesta de OpenAI vacía.")
                return "Lo siento, no pude procesar su solicitud."
            logger.info(f"Respuesta de OpenAI generada: {answer}")
            return answer
        except Exception as e:
            logger.error(f"Generando respuesta con OpenAI, intento {attempts+1}: {e}", exc_info=True)
            attempts += 1
            await asyncio.sleep(1)  # Pequeña pausa antes de reintentar
    return "Lo siento, hubo un error al procesar su solicitud."

def split_text(text, max_length):
    """
    Divide el texto en fragmentos hasta max_length caracteres.
    """
    words = text.split()
    chunks = []
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_length:
            current_chunk += (" " + word) if current_chunk else word
        else:
            chunks.append(current_chunk)
            current_chunk = word
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def concatenate_audio_files(audio_files, output_path):
    """
    Combina múltiples archivos de audio en uno solo.
    """
    try:
        combined = AudioSegment.empty()
        for file in audio_files:
            audio_segment = AudioSegment.from_file(file)
            combined += audio_segment
        combined.export(output_path, format="mp3")
        logger.info(f"Archivos de audio concatenados en: {output_path}")
        for file in audio_files:
            os.remove(file)
            logger.debug(f"Archivo temporal eliminado: {file}")
        return output_path
    except Exception as e:
        logger.error(f"Error al concatenar archivos de audio: {e}", exc_info=True)
        return None

async def generate_audio_async(text, audio_path=None):
    """
    Genera audio usando Eleven Labs en español.
    Divide el texto, genera audio y concatena los fragmentos.
    Si falla, usa errores.mp3 si existe.
    """
    try:
        if not text or len(text.strip()) < 5:
            logger.error("Texto vacío o muy corto para audio.")
            return None
        logger.info(f"Generando audio con Eleven Labs para el texto: {text}")
        text = text[:5000]
        text_chunks = split_text(text, MAX_CHAR_LIMIT)
        logger.debug(f"Texto dividido en {len(text_chunks)} fragmentos.")
        audio_files = []

        for idx, chunk in enumerate(text_chunks):
            logger.debug(f"Generando audio fragmento {idx+1}: {chunk}")
            success = False
            attempts = 0
            while not success and attempts < MAX_RETRIES:
                attempts += 1
                try:
                    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_IDES}"
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
                    }

                    async with httpx.AsyncClient() as client:
                        start_time = time.time()
                        response = await client.post(url, json=data, headers=headers)
                        end_time = time.time()
                        logger.info(f"Intento {attempts}: Tiempo en generar audio frag {idx+1}: {end_time - start_time:.2f} s")

                        if response.status_code == 200:
                            if not os.path.exists("static"):
                                os.makedirs("static")
                            timestamp = int(time.time() * 1000)
                            temp_audio_filename = f"Response_{timestamp}_{idx}.mp3"
                            temp_audio_path = os.path.join("static", temp_audio_filename)
                            with open(temp_audio_path, "wb") as f:
                                f.write(response.content)
                            logger.info(f"Audio frag {idx+1} guardado: {temp_audio_path}")
                            audio_files.append(temp_audio_path)
                            success = True
                        else:
                            logger.error(f"Error frag {idx+1}, intento {attempts}: {response.text}")
                            if attempts >= MAX_RETRIES:
                                logger.error(f"No se pudo generar audio para frag {idx+1} tras {MAX_RETRIES} intentos.")
                    if not success and attempts >= MAX_RETRIES:
                        break
                except Exception as e:
                    logger.error(f"Excepción audio frag {idx+1}, intento {attempts}: {e}", exc_info=True)
                    if attempts >= MAX_RETRIES:
                        logger.error(f"No se pudo generar audio para frag {idx+1} tras {MAX_RETRIES} intentos.")
            if not success:
                break

        if len(audio_files) == len(text_chunks):
            if not audio_files:
                logger.error("No se generaron archivos de audio.")
                return None
            if audio_path is None:
                timestamp = int(time.time())
                audio_filename = f"Response_{timestamp}.mp3"
                audio_path = os.path.join("static", audio_filename)
            final_audio_path = concatenate_audio_files(audio_files, audio_path)
            if final_audio_path:
                logger.info(f"Audio final generado: {final_audio_path}")
                return final_audio_path
            else:
                logger.error("No se pudo concatenar los archivos de audio.")
                return None
        else:
            logger.error("No se pudieron generar todos los fragmentos de audio.")
            if os.path.exists(ERROR_AUDIO_PATH):
                logger.info(f"Usando audio de error: {ERROR_AUDIO_PATH}")
                return ERROR_AUDIO_PATH
            else:
                logger.error("No existe audio de error predeterminado.")
                return None

    except Exception as e:
        logger.error(f"Generando audio asincrónicamente: {e}", exc_info=True)
        return None

def play_audio_to_caller(call_sid, audio_path):
    """
    Reproduce el audio especificado en la llamada.
    Si la llamada no está activa o el audio no existe, se intenta usar el audio de errores.
    """
    try:
        if not call_sid:
            logger.error("call_sid es None, no se puede reproducir el audio.")
            return

        if not is_call_active(call_sid):
            logger.error("La llamada no está activa, no se puede reproducir el audio.")
            return

        if not os.path.exists(audio_path):
            logger.error(f"El archivo de audio no existe: {audio_path}")
            if os.path.exists(ERROR_AUDIO_PATH):
                logger.info(f"Usando audio de errores: {ERROR_AUDIO_PATH}")
                audio_path = ERROR_AUDIO_PATH
            else:
                logger.error("No existe audio de errores, no se puede reproducir nada.")
                return

        audio_url = f"https://{NGROK_URL}/static/{os.path.basename(audio_path)}"
        logger.info(f"URL del audio: {audio_url}")

        response = VoiceResponse()
        response.play(audio_url)
        response.pause(length=60)
        response.redirect(TWIML_BIN_URL)
        logger.debug(f"TwiML generado: {str(response)}")

        twilio_client.calls(call_sid).update(twiml=str(response))
        logger.info("Audio reproducido correctamente al llamante.")
    except Exception as e:
        logger.error(f"Reproduciendo audio: {e}", exc_info=True)

@app.get("/static/{filename}")
async def serve_static(filename: str):
    return FileResponse(path=f"static/{filename}")

@app.websocket("/media")
async def media_socket(websocket: WebSocket):
    """
    WebSocket para Twilio Streaming.
    Recibe eventos 'start', 'media', 'stop' y procesa el audio entrante.
    """
    logger.info("Conexión WebSocket establecida.")
    await websocket.accept()
    audio_queue = queue.Queue()
    stop_event = threading.Event()
    call_context = {
        "call_sid": None,
        "conversation_history": [
            {
                "role": "system",
                "content": (
                    "Eres Jessica, representante de servicio al cliente de NetConnect Servicios de Internet. "
                    "Tu función es ayudar con consultas relacionadas con servicios, facturación, soporte técnico "
                    "e información general. Sigue estrictamente las directrices comerciales."
                ),
            }
        ],
    }

    threading.Thread(
        target=process_audio, args=(audio_queue, stop_event, call_context), daemon=True
    ).start()

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
                    call_context["call_sid"] = event_data["start"]["callSid"]
                    logger.info(f"Call SID recibido: {call_context['call_sid']}")

                    # Usamos bienvenido.mp3 en lugar de welcome.mp3
                    bienvenido_audio_path = os.path.join(STATIC_AUDIO_PATH, "bienvenido.mp3")
                    if os.path.exists(bienvenido_audio_path):
                        play_audio_to_caller(call_context["call_sid"], bienvenido_audio_path)
                    else:
                        logger.error("El archivo de audio estático 'bienvenido' no existe.")

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
                logger.error(f"En WebSocket: {e}", exc_info=True)
                break
    except WebSocketDisconnect:
        logger.info("WebSocket desconectado.")
    finally:
        stop_event.set()
        await websocket.close()
        logger.info("WebSocket cerrado.")

if __name__ == "__main__":
    try:
        logger.info("Iniciando servidor...")
        uvicorn.run(app, host="0.0.0.0", port=5001)
    except Exception as e:
        logger.error(f"Error General: {str(e)}", exc_info=True)
