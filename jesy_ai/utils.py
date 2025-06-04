# jesy_ai/utils.py
import string

def split_text(text, max_length):
    """
    Divide el texto en fragmentos de tamaño menor o igual a max_length.
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

def is_irrelevant(transcript):
    norm = transcript.strip().lower()
    norm = norm.translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())
    irrelevants = [
        "thank you", "ok", "great", "alright", "cool", "uh-huh",
        "i'll wait", "i see", "um", "hmm", "thanks", "okay", "uh huh"
    ]
    return norm in irrelevants

def is_farewell(transcript):
    norm = transcript.strip().lower()
    norm = norm.translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())
    farewells = [
        "goodbye", "bye", "see you", "thank you very much", "thats all i needed"
    ]
    return norm in farewells

def check_quick_intent(transcript, quick_intents):
    norm = transcript.lower().translate(str.maketrans('', '', string.punctuation))
    norm = " ".join(norm.split())
    return quick_intents.get(norm, None)