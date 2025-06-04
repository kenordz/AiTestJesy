import openai
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configurar la clave API
openai.api_key = os.getenv("OPENAI_API_KEY")

def test_openai():
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello Jessica! My internet is not working, can you help me?"}
            ]
        )
        print("Respuesta de OpenAI:", response['choices'][0]['message']['content'])
    except Exception as e:
        print("Error al conectar con la API:", e)

if __name__ == "__main__":
    test_openai()