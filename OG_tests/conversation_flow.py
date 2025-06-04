import openai
import requests
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configurar la API Key de OpenAI
openai.api_key = OPENAI_API_KEY

def generate_response(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input},
        ]
    )
    return response.choices[0].message['content']

def generate_audio(text):
    # Endpoint de ElevenLabs para Text-to-Speech
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "voice_settings": {
            "stability": 1.0,
            "similarity_boost": 0.75
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open("output_audio.wav", "wb") as f:
            f.write(response.content)
        print("Audio guardado como output_audio.wav")
    else:
        print(f"Error en la solicitud: {response.status_code}")
        print(response.json())

def main():
    print("VOXAI Assistant: Hello! How can I assist you today? (type 'exit' to end)")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            print("VOXAI Assistant: Goodbye!")
            break
        response_text = generate_response(user_input)
        print("VOXAI Assistant:", response_text)
        generate_audio(response_text)

if __name__ == "__main__":
    main()