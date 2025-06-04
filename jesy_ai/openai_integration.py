# jesy_ai/openai_integration.py
import time
import openai
import logging

logger = logging.getLogger(__name__)

def get_openai_response_stream_postprocessed(conversation_history):
    """
    Llama a GPT con stream=True y postprocesa el streaming para obtener un único texto final.
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
        logger.error(f"Error in get_openai_response_stream_postprocessed: {e}", exc_info=True)
        return "I'm sorry, I encountered an error while generating my response."

def get_openai_response_sync(conversation_history):
    """
    Llama a GPT-3.5-turbo de manera síncrona.
    """
    try:
        start_time = time.time()
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation_history,
        )
        end_time = time.time()
        logger.info(f"OpenAI tardó {end_time - start_time:.2f}s (sync no-stream).")
        ans = response["choices"][0]["message"]["content"].strip()
        if not ans:
            logger.error("Respuesta vacía de OpenAI (sync no-stream).")
            return "I'm sorry, I couldn't process your request."
        return ans
    except Exception as e:
        logger.error(f"Error in get_openai_response_sync: {e}", exc_info=True)
        return "I'm sorry, I had trouble with my AI engine."

def trim_conversation_history_sync(conversation_history, max_tokens=4000):
    total_tokens = sum(len(msg["content"].split()) for msg in conversation_history)
    while total_tokens > max_tokens and len(conversation_history) > 1:
        removed_msg = conversation_history.pop(1)
        logger.debug(f"Eliminando mensaje antiguo: {removed_msg}")
        total_tokens = sum(len(msg["content"].split()) for msg in conversation_history)
    return conversation_history