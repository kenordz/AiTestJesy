# jesy_ai/twilio_integration.py
import os
import logging
from twilio.twiml.voice_response import VoiceResponse
from config import BASE_URL, TWIML_BIN_URL, twilio_client

logger = logging.getLogger(__name__)

def is_call_active(call_sid):
    try:
        call = twilio_client.calls(call_sid).fetch()
        logger.debug(f"Estado de la llamada: {call.status}")
        return call.status in ["in-progress", "ringing"]
    except Exception as e:
        logger.error(f"Error consultando estado de la llamada: {e}", exc_info=True)
        return False

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
            from config import ERROR_AUDIO_PATH
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