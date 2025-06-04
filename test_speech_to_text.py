from google.cloud import speech

def transcribe_audio():
    client = speech.SpeechClient()

    # Cambia esta ruta al archivo de audio que grabaste
    file_path = "/Users/eugenio/Documents/New Recording 67.wav"

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,  # Asegúrate de que coincida con tu archivo de audio
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        print(f"Transcripción: {result.alternatives[0].transcript}")

if __name__ == "__main__":
    transcribe_audio()
