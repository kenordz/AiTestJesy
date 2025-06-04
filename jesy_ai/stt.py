# jesy_ai/stt.py
import os
import time
import queue
import threading
import logging
import random
from google.cloud import speech

from config import STATIC_AUDIO_PATH, STATIC_MESSAGES, QUICK_INTENTS, ERROR_AUDIO_PATH
from utils import is_irrelevant, is_farewell, check_quick_intent
from openai_integration import get_openai_response_stream_postprocessed, trim_conversation_history_sync
from tts import generate_audio_sync
from twilio_integration import play_audio_to_caller

logger = logging.getLogger(__name__)

def process_audio(audio_queue, stop_event, call_context):
    """
    Hilo que realiza el streaming con Google Speech-to-Text de forma síncrona.
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
            language_code="en-US",
            enable_automatic_punctuation=True,
            max_alternatives=1,
            speech_contexts=[speech.SpeechContext(phrases=[], boost=15.0)],
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
    Maneja la transcripción final de STT:
      - Ignora transcripciones cortas o irrelevantes.
      - Procesa quick intents.
      - Si detecta despedida, reproduce goodbye y detiene.
      - En caso normal, llama a GPT y reproduce la respuesta.
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

                    # 1) Ignorar transcripciones muy cortas o irrelevantes
                    if len(transcript) < 3:
                        logger.info("Transcript corto, ignoramos.")
                        continue
                    if is_irrelevant(transcript):
                        logger.info("Transcript irrelevante, ignoramos.")
                        continue

                    # 2) Verificar si es despedida
                    if is_farewell(transcript):
                        logger.info("Detectada despedida. Reproducimos goodbye y paramos.")
                        goodbye_path = os.path.join(STATIC_AUDIO_PATH, "goodbye.mp3")
                        if os.path.exists(goodbye_path):
                            play_audio_to_caller(call_context["call_sid"], goodbye_path)
                        else:
                            logger.warning("No se encontró goodbye.mp3.")
                        stop_event.set()
                        break

                    # 3) Procesar quick intent
                    quick_answer = check_quick_intent(transcript, QUICK_INTENTS)
                    if quick_answer is not None:
                        logger.info(f"INTENT => {quick_answer}")
                        if quick_answer in STATIC_AUDIO_PATH:  # si se usara para elegir audio estático
                            pass
                        if quick_answer in STATIC_MESSAGES:
                            text_for_intent = STATIC_MESSAGES[quick_answer]
                            quick_audio_path = generate_audio_sync(text_for_intent)
                            if quick_audio_path:
                                play_audio_to_caller(call_context["call_sid"], quick_audio_path)
                            else:
                                logger.error("No se pudo generar TTS para quick_intent msg.")
                        else:
                            # Si es texto directo
                            quick_audio = generate_audio_sync(quick_answer)
                            if quick_audio:
                                play_audio_to_caller(call_context["call_sid"], quick_audio)
                            else:
                                logger.error("No se pudo generar TTS para quick_intent (direct).")
                        continue

                    # 4) Enviar mensaje filler (si aún no se ha enviado)
                    if not wait_message_sent:
                        filler_text = random.choice(STATIC_MESSAGES['wait'])
                        filler_audio = generate_audio_sync(filler_text)
                        if filler_audio:
                            play_audio_to_caller(call_context["call_sid"], filler_audio)
                        else:
                            logger.warning("No se encontró filler audio.")
                        wait_message_sent = True

                    # 5) Agregar transcripción al historial y recortar si es necesario
                    call_context["conversation_history"].append({
                        "role": "user",
                        "content": transcript
                    })
                    call_context["conversation_history"] = trim_conversation_history_sync(
                        call_context["conversation_history"]
                    )

                    # 6) Llamar a GPT con postprocesado streaming
                    logger.info("Llamando GPT con postprocesado streaming (acumular tokens).")
                    gpt_response = get_openai_response_stream_postprocessed(call_context["conversation_history"])
                    logger.info(f"GPT devuelto => {gpt_response}")

                    # Guardar respuesta en el historial
                    call_context["conversation_history"].append({
                        "role": "assistant",
                        "content": gpt_response
                    })

                    # 7) Generar TTS y reproducir la respuesta
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