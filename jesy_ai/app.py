import os
import sys
import difflib
import json
import queue
import base64
import threading
import queue
import time
import random
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
from pydub import AudioSegment
import string

def find_item_in_menu(user_text: str):
    """
    Busca un producto en app.state.super_salads_instructions["MenuSuperSalads"].
    Si lo encuentra, retorna (nombre_producto, precio_float).
    Si no, retorna (None, 0.0).
    """
    if not hasattr(app.state, "super_salads_instructions"):
        return (None, 0.0)  # por si no se cargó
    
    menu_data = app.state.super_salads_instructions["MenuSuperSalads"]
    user_text_lower = user_text.lower()

    # Recorremos cada categoría
    for category, content in menu_data.items():
        if isinstance(content, list):
            # Ej: "Healthy Bakery", "Sopas y Cremas", "Ensaladas"
            for product in content:
                nombre_prod = product.get("nombre", "").lower()
                if nombre_prod in user_text_lower:
                    # 1) CASO sencillo: "precio": "$79"
                    if "precio" in product:
                        price_str = product["precio"].replace("$","").strip()
                        if price_str.isdigit():
                            return (product["nombre"], float(price_str))
                        else:
                            # Manejo fallback
                            return (product["nombre"], 79.0)

                    # 2) CASO con multiples "precios": ["$199","$209","$219"]
                    elif "precios" in product:
                        precios_list = product["precios"]
                        # Por ejemplo:
                        #   indice 0 = solo  => $199
                        #   indice 1 = sopa  => $209
                        #   indice 2 = ensalada => $219
                        if "sopa" in user_text_lower:
                            # Escoger la segunda opción
                            price_str = precios_list[1].replace("$","")
                        elif "ensalada" in user_text_lower:
                            # Tercera opción
                            price_str = precios_list[2].replace("$","")
                        else:
                            # Por defecto la primera
                            price_str = precios_list[0].replace("$","")

                        return (product["nombre"], float(price_str))

        elif isinstance(content, dict):
            # Ej: "Bebidas", "Menú Infantil" (dict en vez de lista)
            for k, v in content.items():
                k_lower = k.lower()
                if k_lower in user_text_lower:
                    # v = {"precio":"$59"} por ej
                    price_str = v.get("precio","").replace("$","").strip()
                    if price_str.isdigit():
                        return (k, float(price_str))
                    else:
                        # fallback
                        return (k, 59.0)

        # Aquí puedes meter más elif si hay más formatos

    # Si no encontramos nada, devuelves (None, 0.0)
    return (None, 0.0)
def debug_call_instance(call_sid):
    """
    Esta función es SOLO para depurar: imprime todos los atributos
    y valores que contenga el objeto 'call'.
    """
    try:
        call = twilio_client.calls(call_sid).fetch()
        logger.info(f"[debug_call_instance] call.__dict__: {call.__dict__}")

        # Si ves que aparece algo como '_record' o '_payload' en el dict, imprímelo también:
        if "_payload" in call.__dict__:
            logger.info(f"[debug_call_instance] call._payload: {call._payload}")
        if "_record" in call.__dict__:
            logger.info(f"[debug_call_instance] call._record: {call._record}")

    except Exception as e:
        logger.error(f"[debug_call_instance] Error consultando call_sid={call_sid}: {e}", exc_info=True)
# ---------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Cargar variables de entorno desde .env
# ---------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------
# Validar variables obligatorias
# ---------------------------------------------------------------------
required_env_vars = [
    "ELEVENLABS_API_KEY",
    "VOICE_ID_ES",
    "TWILIO_ACCOUNT_SID",
    "TWILIO_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "TWIML_BIN_URL",
]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.error(f"Faltan variables de entorno: {', '.join(missing_vars)}")
    sys.exit(1)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID_ES = os.getenv("VOICE_ID_ES")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWIML_BIN_URL = os.getenv("TWIML_BIN_URL")

# ---------------------------------------------------------------------
# URL pública de Cloud Run
# ---------------------------------------------------------------------
BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://ai-test-app-97012651308.us-central1.run.app")

openai.api_key = OPENAI_API_KEY

# ---------------------------------------------------------------------
# Inicializar FastAPI y Twilio
# ---------------------------------------------------------------------
app = FastAPI()
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# ---------------------------------------------------------------------
# Parámetros Eleven Labs (TTS)
# ---------------------------------------------------------------------
ELEVENLABS_STABILITY = 0.1
ELEVENLABS_SIMILARITY_BOOST = 0.9
MAX_CHAR_LIMIT = 600
MAX_RETRIES = 3

# ---------------------------------------------------------------------
# Rutas de archivos estáticos
# ---------------------------------------------------------------------
STATIC_AUDIO_PATH = "static"
ERROR_AUDIO_PATH = os.path.join(STATIC_AUDIO_PATH, "error.mp3")

# ---------------------------------------------------------------------
# Mensajes estáticos
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

    # AGREGAR NUEVO: claves para quick_intents fijos
    "hola_me_puedes_ayudar": "¡Hola! Con gusto, ¿en qué puedo ayudarte hoy?",
    "cual_es_tu_nombre": "Soy Jessica, tu asistente virtual de SuperSalads.",
    "eres_una_persona_real": "Soy una asistente virtual, pero estoy aquí para atenderte como si fuera en persona.",
    "farewell_quick": "¡De nada! Muchas gracias por llamar a SuperSalads. ¡Que tengas un excelente día! Hasta pronto.",

    # Ejemplos específicos para restaurante
    "domicilio_info": "Tenemos servicio a domicilio y para llevar. ¿Te gustaría conocer nuestras opciones?",
    "support_hours": "Nuestros horarios son de lunes a domingo, 12pm a 11pm.",
}

# ---------------------------------------------------------------------
# Frases de despedida
# ---------------------------------------------------------------------
farewell_phrases = [
   "adios",
    "hasta luego",
    "muchas gracias",
    "nos vemos",
    "eso era todo"
]

# ---------------------------------------------------------------------
# quick_intents (mini-tree)
#  En lugar de texto “directo”, referimos keys de STATIC_MESSAGES para TTS
# ---------------------------------------------------------------------
QUICK_INTENTS = {
    "hola me puedes ayudar": "hola_me_puedes_ayudar",
    "cual es tu nombre": "cual_es_tu_nombre",
    "eres una persona real": "eres_una_persona_real",

    # Ejemplos específicos para restaurante
    "tienen servicio a domicilio": "domicilio_info",
    "cuales son los horarios": "support_hours",
    "adios": "farewell_quick",
    "hasta luego": "farewell_quick",
    "Seria todo muchas gracias": "farewell_quick",
    # Intención para enviar ubicación vía SMS
    "mandame la ubicacion": "send_location_sms",
    "quiero hacer un pedido a domicilio": "send_menu_sms",
    "quisiera pedirte a domicilio": "send_menu_sms",

    #Finalizar pedido
    "finalizar pedido": "close_order",
    "ya es todo": "close_order"
}
# ---------------------------------------------------------------------
# split_text para trocear texto largo en TTS
# ---------------------------------------------------------------------
def split_text(text, max_length):
    """
    Divide el texto en chunks <= max_length (en caracteres).
    """
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
# Endpoint de prueba
# ---------------------------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
# ---------------------------------------------------------------------
# función fuzzy_check_quick_intent()
# ---------------------------------------------------------------------

def fuzzy_check_quick_intent(transcript, cutoff=0.6):
    """
    Hace coincidencia aproximada entre lo que dijo el usuario (transcript)
    y las claves de QUICK_INTENTS. Si la similitud es >= cutoff,
    retorna la intención. Si no, None.
    """
    # Normalizamos texto para que sea consistente (todo minúsculas, sin puntuación)
    norm = transcript.lower().translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())

    # Lista de las frases clave definidas en QUICK_INTENTS
    possible_phrases = list(QUICK_INTENTS.keys())

    # Usamos get_close_matches para encontrar la frase "más parecida"
    matches = difflib.get_close_matches(norm, possible_phrases, n=1, cutoff=cutoff)
    if matches:
        # 'matches[0]' será la frase de QUICK_INTENTS más similar a 'norm'
        matched_key = matches[0]
        # Devolvemos la intención (el valor en QUICK_INTENTS)
        return QUICK_INTENTS[matched_key]
    else:
        return None
# ---------------------------------------------------------------------
# Generar audios estáticos al inicio (startup)
# ---------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    # 1) Cargar las instrucciones desde el JSON
    try:
        with open("instrucciones_super_salads.json", "r", encoding="utf-8") as f:
            data_instrucciones = json.load(f)
        # Guardamos en app.state
        app.state.super_salads_instructions = data_instrucciones
        logger.info("Instrucciones de SuperSalads cargadas correctamente.")
    except Exception as e:
        logger.error(f"Error al cargar instrucciones_super_salads.json: {e}", exc_info=True)
    await generate_static_audio()

async def generate_static_audio():
    """
    Genera audios estáticos (bienvenido, espera, errores, etc.) si no existen
    al inicio.
    """
    if not os.path.exists(STATIC_AUDIO_PATH):
        os.makedirs(STATIC_AUDIO_PATH)

    # Generar audios de "wait" y demás
    for key, text in STATIC_MESSAGES.items():
        # 'wait' es una lista de frases
        if key == 'espera':
            for idx, phrase in enumerate(text):
                audio_filename = f"{key}_{idx}.mp3"
                audio_path = os.path.join(STATIC_AUDIO_PATH, audio_filename)
                if not os.path.exists(audio_path):
                    logger.info(f"Generando audio estático: {audio_path}")
                    await generate_audio_file_static(phrase, audio_path)
                else:
                    logger.info(f"Audio estático ya existe: {audio_path}")
        else:
            audio_filename = f"{key}.mp3"
            audio_path = os.path.join(STATIC_AUDIO_PATH, audio_filename)
            if not os.path.exists(audio_path):
                logger.info(f"Generando audio estático: {audio_path}")
                await generate_audio_file_static(text, audio_path)
            else:
                logger.info(f"Audio estático ya existe: {audio_path}")

    # Verificar goodbye
    goodbye_audio_path = os.path.join(STATIC_AUDIO_PATH, "goodbye.mp3")
    if not os.path.exists(goodbye_audio_path):
        logger.warning("No existe goodbye.mp3; no se podrá reproducir despedida.")
    else:
        logger.info("Archivo goodbye.mp3 disponible.")

async def generate_audio_file_static(text, audio_path):
    logger.info(f"Generando audio estático para: {text}")
    final_path = await generate_audio_async(text, audio_path)
    if final_path:
        logger.info(f"Audio estático guardado: {final_path}")
    else:
        logger.error("No se pudo generar audio estático.")

# ---------------------------------------------------------------------
# HILO PARA PROCESAR AUDIO EN TIEMPO REAL (Google STT)
# ---------------------------------------------------------------------
def process_audio(audio_queue, stop_event, call_context):
    """
    Hilo que hace streaming con Google STT (síncrono).
    """
    try:
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            logger.warning("Removiendo GOOGLE_APPLICATION_CREDENTIALS a la fuerza...")
            del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        logger.info("Inicializando cliente de Google Speech-to-Text (sync).")
        speech_client = speech.SpeechClient()

        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
            sample_rate_hertz=8000,
            language_code="es-MX",
            enable_automatic_punctuation=True,
            max_alternatives=2,
            use_enhanced=True,
            model="phone_call",
            speech_contexts=[speech.SpeechContext(
                 phrases=[
                    "SuperSalads", "Gómez Morín", "Menú Infantil",
                    "ensalada", "domicilio", "facturación", 
                    "paquete", "entregar", "sopa", "crema",
                    "promoción", "tarjeta", "efectivo", 
                    "Healthy Bakery", "postre", "bebida",
                    "hamburguesa", "vegetariano", "cupón", "domicilio"
                 ],
                 boost=20.0  # ↑ Prioridad para estas palabras
            )],
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
                        buffer = bytearray()
                except queue.Empty:
                    if (time.time() - last_audio_time) >= 0.03:
                        logger.debug("Enviando silencio a STT.")
                        silence = b"\xFF" * 320
                        yield speech.StreamingRecognizeRequest(audio_content=silence)
                        last_audio_time = time.time()

        requests_generator = audio_generator()
        logger.info("Iniciando streaming_recognize con Google STT...")
        responses = speech_client.streaming_recognize(streaming_config, requests_generator)
        logger.info("STT en marcha; iremos a handle_responses_sync.")
        handle_responses_sync(responses, call_context, stop_event)

    except Exception as e:
        logger.error(f"Error procesando audio: {e}", exc_info=True)


def handle_responses_sync(responses, call_context, stop_event):
    """
    instructions = app.state.super_salads_instructions
    Maneja la transcripción final de STT:
      - Si irrelevante, ignoramos.
      - Si quick intent, sacamos su audio estático (o texto).
      - Si despedida, goodbye y paramos.
      - Si normal, filler + GPT => TTS => reproducir
    """
    try:
        wait_message_sent = False
        for response in responses:
            if not response.results:
                continue
            for result in response.results:
                if result.is_final:
                    transcript = result.alternatives[0].transcript.strip()
                    logger.info(f"Transcripción final: {transcript}")

                    # 1) ignorar corto o irrelevante
                    if len(transcript) < 3:
                        logger.info("Transcript corto, ignoramos.")
                        continue
                    if is_irrelevant(transcript):
                        logger.info("Transcript irrelevante, ignoramos.")
                        continue

                    # 2) check farewell
                    if is_farewell(transcript):
                        logger.info("Detectada despedida. Reproducimos goodbye y paramos.")
                        goodbye_path = os.path.join(STATIC_AUDIO_PATH, "goodbye.mp3")
                        if os.path.exists(goodbye_path):
                            play_audio_to_caller(call_context["call_sid"], goodbye_path)
                        else:
                            logger.warning("No se encontró goodbye.mp3.")
                        stop_event.set()
                        break

                    # 3) quick intent
                    quick_answer = fuzzy_check_quick_intent(transcript)
                    if quick_answer is not None:
                        logger.info(f"INTENT => {quick_answer}")

                        # NUEVO: Si es "send_location_sms", enviamos SMS
                        if quick_answer == "send_location_sms":  # NUEVO
                            location_tts = "Sure, I'm sending the location to your phone now."
                            quick_audio_path = generate_audio_sync(location_tts)
                            if quick_audio_path:
                                play_audio_to_caller(call_context["call_sid"], quick_audio_path)
                            else:
                                logger.error("No se pudo generar TTS para 'send_location_sms' intent.")
                            
                            # Enviar SMS al número que llama
                            send_sms_to_caller(
                                call_context["call_sid"],
                                "Hola, soy Jessica! Te comparto la ubicación de SUPERSALADS: Gómez Morín 81 8335 1245 / 81 8335 3779 Link de maps: https://maps.app.goo.gl/4U5gqecr2PoA9NXj6"
                            )
                            continue  # Saltamos GPT

                        # NUEVO: Manejo de "send_menu_sms"
                        elif quick_answer == "send_menu_sms":
                            static_audio = os.path.join(STATIC_AUDIO_PATH, "send_menu_sms.mp3")
                            if os.path.exists(static_audio):
                                play_audio_to_caller(call_context["call_sid"], static_audio)
                                
                            else:
                                logger.error("No se encontró send_menu_sms.mp3.")
        
                            # Envía SMS con el link a tu PDF
                            send_sms_to_caller(
                                call_context["call_sid"],
                                "¡Hola! Aquí tienes el menú de SuperSalads: https://www.supersaladsmx.site/men%C3%BA-domicilio-mty"
                            )
                            continue  # Saltamos GPT
                        # Manejo de CLOSE ORDER
                        elif quick_answer == "close_order":
                            # 1) Generar número de pedido (al azar, 4 dígitos)
                            order_id = random.randint(1000, 9999)

                            # 2) Tomar el total que llevas en call_context["order_subtotal"]
                            total = call_context["order_subtotal"]

                            # 3) Decirle al usuario el total y el número de pedido
                            summary = f"Tu total es {total} pesos. Tu número de pedido es {order_id}. En 40 minutos llega a tu domicilio."
                            path_sum = generate_audio_sync(summary)
                            if path_sum:
                                play_audio_to_caller(call_context["call_sid"], path_sum)

                            # 4) (Opcional) Reiniciar el pedido
                            call_context["current_order"].clear()
                            call_context["order_subtotal"] = 0.0

                            # 5) Continúa o cierra llamada
                            # stop_event.set()  # si quisieras colgar
                            
                            continue
    
                        # Si quick_answer es string que apunta a STATIC_MESSAGES
                        if quick_answer in STATIC_MESSAGES:
                            audio_filename = f"{quick_answer}.mp3"  # e.g. "hola_me_puedes_ayudar.mp3"
                            audio_path = os.path.join(STATIC_AUDIO_PATH, audio_filename)
                            if os.path.exists(audio_path):
                                play_audio_to_caller(call_context["call_sid"], audio_path)
                            else:
                                logger.error(f"No se encontró audio estático para la clave {quick_answer}")
                            continue
                        else:
                            # quick_answer es texto directo
                            quick_audio = generate_audio_sync(quick_answer)
                            if quick_audio:
                                play_audio_to_caller(call_context["call_sid"], quick_audio)
                            else:
                                logger.error("No se pudo generar TTS para quick_intent (direct).")
                        continue  # sin GPT
                    # - - -  DETECTAR PRODUCTO - - -
                    nombre_encontrado, precio_encontrado = find_item_in_menu(transcript)
                    if nombre_encontrado:
                        # Agregamos el item al "carrito"
                        call_context["current_order"].append({
                            "item": nombre_encontrado,
                            "price": precio_encontrado
                        })
                        call_context["order_subtotal"] += precio_encontrado

                        # Generar TTS diciendo que se agregó
                        tts_text = f"Agregué {nombre_encontrado} con precio de {precio_encontrado} pesos. ¿Algo más?"
                        tts_path = generate_audio_sync(tts_text)
                        if tts_path:
                            play_audio_to_caller(call_context["call_sid"], tts_path)

                        continue

                    # 4) filler (si no mandamos ya)
                    if not wait_message_sent:
                        idx = random.randint(0, len(STATIC_MESSAGES["espera"]) - 1)
                        audio_filename = f"espera_{idx}.mp3"  # e.g. "espera_0.mp3"
                        audio_path = os.path.join(STATIC_AUDIO_PATH, audio_filename)
                        if os.path.exists(audio_path):
                            play_audio_to_caller(call_context["call_sid"], audio_path)
                        else:
                            logger.warning(f"No se encontró el MP3 {audio_filename} de 'espera'.")
                        wait_message_sent = True

                    # Añadir user transcripción al historial
                    call_context["conversation_history"].append({
                        "role": "user",
                        "content": transcript
                    })
                    call_context["conversation_history"] = trim_conversation_history_sync(
                        call_context["conversation_history"]
                    )

                    # Llamar GPT con postprocesamiento streaming (opción B)
                    logger.info("Llamando GPT con postprocesado streaming (acumular tokens).")
                    gpt_response = get_openai_response_stream_postprocessed(call_context["conversation_history"])
                    logger.info(f"GPT devuelto => {gpt_response}")

                    # Guardar en historial
                    call_context["conversation_history"].append({
                        "role": "assistant",
                        "content": gpt_response
                    })

                    # Generar TTS en un solo mp3
                    audio_gpt = generate_audio_sync(gpt_response)
                    if audio_gpt:
                        play_audio_to_caller(call_context["call_sid"], audio_gpt)
                    else:
                        logger.error("No se pudo generar TTS de GPT.")
                        if os.path.exists(ERROR_AUDIO_PATH):
                            play_audio_to_caller(call_context["call_sid"], ERROR_AUDIO_PATH)

                    wait_message_sent = False

                    if stop_event.is_set():
                        break
            if stop_event.is_set():
                break
    except Exception as e:
        logger.error(f"Error en handle_responses_sync: {e}", exc_info=True)

# ---------------------------------------------------------------------
# NUEVO: Función para enviar SMS al número que llama
# ---------------------------------------------------------------------
def send_sms_to_caller(call_sid, message):
    try:
        call = twilio_client.calls(call_sid).fetch()

        # Observado en logs: '_from' está en call.__dict__
        from_number = call.__dict__["_from"]
        
        # Si tu número local no comienza con '+', antepón '+52'
        if not from_number.startswith("+"):
            from_number = f"+52{from_number}"

        # Envía el SMS usando tu número de Twilio
        twilio_client.messages.create(
            body=message,
            from_="+19253293387",  # tu número Twilio
            to=from_number
        )
        logger.info(f"SMS enviado a {from_number}")

    except Exception as e:
        logger.error(f"Error enviando SMS: {e}", exc_info=True)

# ---------------------------------------------------------------------
# get_openai_response_stream_postprocessed
# (Opción B: postprocesar streaming => single text final)
# ---------------------------------------------------------------------
def get_openai_response_stream_postprocessed(conversation_history):
    """
    Llama a GPT con stream=True, NO reproducimos parcial. 
    Juntamos todo en un 'full_text' y lo devolvemos al final.
    """
    try:
        logger.info("[get_openai_response_stream_postprocessed] => stream=True (post-proc).")
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
            stream=True
        )
        full_text = ""
        for chunk in resp:
            if "choices" not in chunk:
                continue
            if not chunk["choices"]:
                continue
            delta = chunk["choices"][0]["delta"]
            content_part = delta.get("content", "")
            full_text += content_part
        final = full_text.strip()
        return final if final else "I'm sorry, I had trouble finalizing my response."
    except Exception as e:
        logger.error(f"Error en get_openai_response_stream_postprocessed: {e}", exc_info=True)
        return "I'm sorry, I encountered an error while generating my response."

# ---------------------------------------------------------------------
# Llamada GPT no-stream (legacy). Mantenemos para referencia
# ---------------------------------------------------------------------
def get_openai_response_sync(conversation_history):
    """
    Llamada a GPT-3.5-turbo (sync).
    """
    try:
        start_ = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
        )
        end_ = time.time()
        logger.info(f"OpenAI tardó {end_ - start_:.2f}s (sync no-stream).")
        ans = response["choices"][0]["message"]["content"].strip()
        if not ans:
            logger.error("Respuesta vacía de OpenAI (sync no-stream).")
            return "I'm sorry, I couldn't process your request."
        return ans
    except Exception as e:
        logger.error(f"Error en get_openai_response_sync: {e}", exc_info=True)
        return "I'm sorry, I had trouble with my AI engine."

# ---------------------------------------------------------------------
# Recortar historial GPT si excede tokens
# ---------------------------------------------------------------------
def trim_conversation_history_sync(conversation_history, max_tokens=4000):
    total_tokens = sum(len(msg["content"].split()) for msg in conversation_history)
    while total_tokens > max_tokens and len(conversation_history) > 1:
        removed_msg = conversation_history.pop(1)
        logger.debug(f"Eliminando mensaje antiguo: {removed_msg}")
        total_tokens = sum(len(msg["content"].split()) for msg in conversation_history)
    return conversation_history

# ---------------------------------------------------------------------
# Generar TTS con Eleven Labs (sync)
# ---------------------------------------------------------------------
def generate_audio_sync(text):
    """
    Genera TTS con Eleven Labs (sync).
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
            # Añade este campo:
            "model_id": "eleven_multilingual_v1"
                }
                start_time = time.time()
                resp = httpx.post(url, json=data, headers=headers)
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

# ---------------------------------------------------------------------
# Lógica asíncrona para audios estáticos
# ---------------------------------------------------------------------
async def generate_audio_async(text, audio_path=None):
    """
    Generar audios estáticos en arranque (ej: bienvenido), asíncrono.
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
            # Añade este campo:
            "model_id": "eleven_multilingual_v1"
                        }
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

# ---------------------------------------------------------------------
# Validar si la llamada Twilio sigue activa
# ---------------------------------------------------------------------
def is_call_active(call_sid):
    try:
        call = twilio_client.calls(call_sid).fetch()
        logger.debug(f"Estado de la llamada: {call.status}")
        return call.status in ["in-progress", "ringing"]
    except Exception as e:
        logger.error(f"Error consultando estado de la llamada: {e}", exc_info=True)
        return False

# ---------------------------------------------------------------------
# Checar si es irrelevante
# ---------------------------------------------------------------------
def is_irrelevant(transcript):
    norm = transcript.strip().lower()
    norm = norm.translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())
    irrelevants = [
        "gracias","ok","perfecto","okay","cool","uh-huh",
        "i'll wait","i see","um","hmm","entiendo","okay","uh huh"
    ]
    return norm in irrelevants

# ---------------------------------------------------------------------
# Checar si es despedida
# ---------------------------------------------------------------------
def is_farewell(transcript):
    norm = transcript.strip().lower()
    norm = norm.translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())
    farewells = [
        "adios","hasta luego","muchas gracias","nos vemos","hasta pronto","eso era todo"
    ]
    return norm in farewells

# ---------------------------------------------------------------------
# Definir check_quick_intent: compara con QUICK_INTENTS
# ---------------------------------------------------------------------
def check_quick_intent(transcript):
    norm = transcript.lower().translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())
    if norm in QUICK_INTENTS:
        return QUICK_INTENTS[norm]
    return None

# ---------------------------------------------------------------------
# Reproducir audio con Twilio <Play> + <Redirect>
# ---------------------------------------------------------------------
def play_audio_to_caller(call_sid, audio_path):
    try:
        if not call_sid:
            logger.error("call_sid es None, no se puede reproducir audio.")
            return
        if not is_call_active(call_sid):
            logger.error("La llamada no está activa, no se puede reproducir audio.")
            return

        if not os.path.exists(audio_path):
            logger.error(f"No existe archivo de audio: {audio_path}")
            if os.path.exists(ERROR_AUDIO_PATH):
                audio_path = ERROR_AUDIO_PATH
            else:
                logger.error("No se encontró error.mp3. No se reproduce nada.")
                return

        audio_url = f"{BASE_URL}/static/{os.path.basename(audio_path)}"
        logger.info(f"Twilio Play => {audio_url}")

        resp = VoiceResponse()
        resp.play(audio_url)
        resp.pause(length=60)
        resp.redirect(TWIML_BIN_URL)

        twilio_client.calls(call_sid).update(twiml=str(resp))
        logger.info("Audio reproducido vía Twilio correctamente.")
    except Exception as e:
        logger.error(f"Error reproduciendo audio: {e}", exc_info=True)

# ---------------------------------------------------------------------
# Endpoint para servir /static/<filename>
# ---------------------------------------------------------------------
@app.get("/static/{filename}")
async def serve_static(filename: str):
    return FileResponse(path=f"static/{filename}")

# ---------------------------------------------------------------------
# WebSocket /media con Twilio
# ---------------------------------------------------------------------
import asyncio
import json
import threading
import logging
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse


# -----------
#  APP WEBSOCKETS (ejemplo)
# -----------

@app.websocket("/media")
async def media_socket(websocket: WebSocket):
    logger = logging.getLogger(__name__)
    logger.info("Conexión WebSocket /media establecida con Twilio.")
    await websocket.accept()

    audio_queue = queue.Queue()
    stop_event = threading.Event()
    call_context = {
        "call_sid": None,
        "conversation_history": [
            {
                "role": "system",
                "content": """Eres Jessica, una asistente virtual para SuperSalads, especializada en atención al cliente.
Habla en español y ayudas con información sobre el menú, ubicaciones, horarios, pedidos a domicilio y cualquier duda.
Eres amable y sigues las políticas de la empresa.
Siempre responde en español.
Siempre respondes los numeros en español.
Por favor, da respuestas concisas, con un máximo de 30 palabras.

AQUÍ TIENES EL MENÚ OFICIAL EN PESOS (JSON).
No inventes datos que no estén aquí.
Si algo no está en este JSON, di que no lo sabes.
No uses otras monedas mas que Pesos.
"Solo utiliza la información disponible en app.state.super_salads_instructions. "
 },
      "opciones_pago": {
        "descripcion": "Debe informar sobre los métodos de pago disponibles.",
        "opciones": [
          "Efectivo",
          "Tarjeta"
        ],
        "calculo_cambio": "Si el cliente paga en efectivo, calcular el cambio que el repartidor debe llevar."
      },
      "confirmacion_final": {
        "descripcion": "Antes de finalizar, debe confirmar todos los datos con el cliente.",
        "confirmar_datos": [
          "Pedido",
          "Dirección (si aplica)",
          "Método de pago",
          "Promociones vigentes (si aplica)"
        ]
      },
      "pedidos_domicilio": {
        "descripcion": "Para pedidos a domicilio, debe solicitar información completa del cliente.",
        "campos_obligatorios": [
          "Nombre",
          "Dirección",
          "Número de teléfono",
          "Entre calles (solo si aplica)"
        ]
      },
      "pedidos_recoger": {
        "descripcion": "Para pedidos para recoger, solo se debe solicitar el nombre y número de teléfono.",
        "no_pedir_direccion": true
      },
      "combos": {
        "descripcion": "Si el cliente pide un combo, preguntar si lo quiere en combo o individual.",
        "preguntar": [
          "Tipo de pan",
          "Opciones disponibles"
        ],
        "reglas_precios": {
          "no_mencionar_precio_individual": true,
          "solo_mencionar_total": "A menos que el cliente lo pida"
        }
      }
    },
    "restricciones": {
      "panes_paninis": {
        "descripcion": "Cada panini solo puede pedirse con los panes disponibles en su sección.",
        "ejemplo_panini_palermo": {
          "panes_disponibles": [
            "Centeno",
            "Ciabatta Multigrano"
          ],
          "panes_no_permitidos": [
            "Pan blanco",
            "Pan brioche",
            "Pan pita"
          ]
        }
      },
      "sustituciones_permitidas": {
        "descripcion": "Se pueden cambiar ingredientes dentro de las opciones disponibles en el menú.",
        "ejemplo_santa_fe_spicy": {
          "aderezos_disponibles": [
            "Ranch Spicy",
            "Ranch",
            "Blue Cheese"
          ],
          "sustituciones_no_permitidas": [
            "Aderezo de Mango Sriracha",
            "Aderezo de Mostaza"
          ]
        }
      },
      "cargo_envio": {
        "descripcion": "El cargo de envío es de 40 pesos y solo se menciona si el cliente lo pregunta o al final del pedido.",
        "monto": 40
      },
      "areas_servicio": {
        "descripcion": "Cada sucursal solo entrega a domicilio dentro de un rango de 8 km.",
        "restriccion": "Si el cliente está fuera del área de entrega, se le ofrece la opción de recoger en la sucursal más cercana."
      }


"""
            }
        ],
        "current_order": [],
        "order_subtotal": 0.0
    }

    # Lanzar el hilo para procesar audio en tiempo real
    threading.Thread(
        target=process_audio,
        args=(audio_queue, stop_event, call_context),
        daemon=True
    ).start()

    try:
        while not stop_event.is_set():
            try:
                message = await websocket.receive_text()
                if message is None:
                    logger.info("WebSocket cerrado por el cliente.")
                    break
                data = json.loads(message)
                event = data.get("event", "")
                logger.debug(f"Evento WebSocket: {event}")

                if event == "start":
                    call_context["call_sid"] = data["start"]["callSid"]
                    logger.info(f"Call SID recibido: {call_context['call_sid']}")
                    # Reproducir bienvenido.mp3
                    bienvenido_path = os.path.join(STATIC_AUDIO_PATH, "bienvenido.mp3")
                    if os.path.exists(bienvenido_path):
                        play_audio_to_caller(call_context["call_sid"], bienvenido_path)
                    else:
                        logger.error("No se encontró bienvenido.mp3.")

                elif event == "media":
                    payload = data["media"]["payload"]
                    audio_content = base64.b64decode(payload)
                    audio_queue.put(audio_content)
                    logger.debug(f"Se puso audio en cola: {len(audio_content)} bytes")

                elif event == "stop":
                    logger.info("Evento 'stop' recibido. Cerrando WebSocket.")
                    stop_event.set()
                    break
                else:
                    logger.warning(f"Evento no manejado: {event}")
            except Exception as e:
                logger.error(f"Excepción en WebSocket: {e}", exc_info=True)
                break
    except WebSocketDisconnect:
        logger.info("WebSocket desconectado por el cliente.")
    finally:
        stop_event.set()
        await websocket.close()
        logger.info("WebSocket /media cerrado al final.")

# ---------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8080))
        logger.info(f"Iniciando servidor en el puerto {port}...")
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        logger.error(f"Error General: {str(e)}", exc_info=True)

        