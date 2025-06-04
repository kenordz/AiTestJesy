from gevent import monkey
monkey.patch_all()

import os
import json
import base64
import threading
from flask import Flask, request, send_from_directory
from twilio.twiml.voice_response import VoiceResponse
from dotenv import load_dotenv
import requests
import openai
from google.cloud import speech_v1p1beta1 as speech
from twilio.rest import Client
from flask_sockets import Sockets
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler

# Load environment variables from the .env file
load_dotenv()

# Set environment variable for Google Application Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Configure API keys and credentials
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("VOICE_ID")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NGROK_URL = os.getenv("NGROK_URL")  # Ensure this is set in your .env file

# Configure OpenAI
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
sockets = Sockets(app)

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Function to generate a response from OpenAI
def get_openai_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are Monica, a helpful assistant for NetConnect Services."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error generating response with OpenAI: {e}")
        return "I'm sorry, I couldn't process that."

# Function to generate an audio file with Eleven Labs
def generate_audio(text):
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
        if not os.path.exists("static"):
            os.makedirs("static")
        with open("static/Response.wav", "wb") as f:
            f.write(response.content)
        print("Audio saved as static/Response.wav")

        # Convert audio to 8 kHz and mono using ffmpeg
        os.system("ffmpeg -i static/Response.wav -ar 8000 -ac 1 static/Response_Converted.wav -y")
        print("Converted audio saved as static/Response_Converted.wav")
        return "static/Response_Converted.wav"
    else:
        print(f"Error in request: {response.status_code}")
        print(response.json())
        return None

# Endpoint to handle the call and start the media stream
@app.route("/voice", methods=["POST"])
def voice():
    response = VoiceResponse()
    response.say("Thank you for calling NetConnect Services. How can I assist you today?")
    
    # Start the media stream
    start = response.start()
    start.stream(url=f"wss://{NGROK_URL}/media")
    
    return str(response)

@sockets.route('/media')
def media_socket(ws):
    print("WebSocket connection established")
    call_sid = None

    # Configure Google Cloud Speech recognition config
    speech_client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        language_code="en-US",
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False
    )

    def generator():
        nonlocal call_sid
        while True:
            message = ws.receive()
            if message is None:
                break
            data = json.loads(message)
            event = data.get('event')
            if event == 'start':
                call_sid = data['start']['callSid']
                print(f"Call SID: {call_sid}")
            elif event == 'media':
                payload = data['media']['payload']
                audio_content = base64.b64decode(payload)
                yield speech.StreamingRecognizeRequest(audio_content=audio_content)
            elif event == 'stop':
                print("Media stream stopped")
                break

    requests_generator = generator()
    responses = speech_client.streaming_recognize(config=streaming_config, requests=requests_generator)

    # Process the responses in a separate thread
    threading.Thread(target=process_responses, args=(responses, call_sid)).start()

def process_responses(responses, call_sid):
    for response in responses:
        for result in response.results:
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print(f"Transcript: {transcript}")

                # Generate response from OpenAI
                openai_response = get_openai_response(transcript)
                print(f"OpenAI Response: {openai_response}")

                # Generate audio response
                response_audio_path = generate_audio(openai_response)
                if response_audio_path:
                    # Play the response back to the caller
                    play_audio_to_caller(call_sid, response_audio_path)
                else:
                    print("Error generating audio response")
                return  # Exit after processing the final result

def play_audio_to_caller(call_sid, audio_path):
    # Use Twilio's API to send a <Play> verb to the caller
    twiml_response = VoiceResponse()
    audio_url = request.url_root.rstrip('/') + '/' + audio_path
    twiml_response.play(audio_url)
    print(f"Playing audio to caller: {audio_url}")

    twilio_client.calls(call_sid).update(twiml=str(twiml_response))

# Route to serve static files (if needed)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    print("Starting server...")
    server = pywsgi.WSGIServer(('0.0.0.0', 5001), app, handler_class=WebSocketHandler)
    server.serve_forever()