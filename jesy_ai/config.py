# jesy_ai/config.py
import os
import sys
import logging
from dotenv import load_dotenv

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno desde .env
load_dotenv()

# Validar variables obligatorias
required_env_vars = [
    "ELEVENLABS_API_KEY",
    "VOICE_ID",
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
VOICE_ID = os.getenv("VOICE_ID")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TWIML_BIN_URL = os.getenv("TWIML_BIN_URL")

# URL pública de Cloud Run (o la que corresponda)
BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://ai-test-app-97012651308.us-central1.run.app")

# Configuración de OpenAI
import openai
openai.api_key = OPENAI_API_KEY

# Inicializar Twilio Client
from twilio.rest import Client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Rutas de archivos estáticos
STATIC_AUDIO_PATH = "static"
ERROR_AUDIO_PATH = os.path.join(STATIC_AUDIO_PATH, "error.mp3")

# Parámetros para Eleven Labs (TTS)
ELEVENLABS_STABILITY = 0.1
ELEVENLABS_SIMILARITY_BOOST = 0.9
MAX_CHAR_LIMIT = 300
MAX_RETRIES = 3

# Mensajes estáticos
STATIC_MESSAGES = {
    "welcome": "Thank you for connecting to NetConnect Services, how can I help you today?",
    "wait": [
        "Ok, give me one second, please.",
        "Ok, one moment, please.",
        "Hold on, please.",
        "Just a moment, please.",
        "Alright, let me check that for you, one second."
    ],
    "error": "We are experiencing technical difficulties. Please try again later.",
    "slow_internet": (
        "I'm sorry to hear that your internet is running slow. Let me walk you through some quick steps. "
        "Have you tried restarting your modem by unplugging it for 10 seconds and plugging it back in?"
    ),
    "payment_due": (
        "Your billing cycle ends on the 10th each month, and payment is due by the 15th. "
        "Currently, your balance is 45 dollars. Would you like information on payment methods?"
    ),
    "support_hours": (
        "Jesy AI is available 24/7 for automated assistance. Our human support team is available Monday "
        "to Friday from 9 AM to 6 PM, and on Saturdays from 10 AM to 2 PM."
    ),
    "issue_resolved": (
        "I'm glad I could help you fix that. If there's anything else you need, feel free to let me know. "
        "Have a wonderful day!"
    ),
    "implementation": (
        "Jesy AI can be integrated into virtually any industry that relies on customer service, such as "
        "airlines, restaurants, hotels, internet providers, and many more. It's designed to streamline support "
        "and improve customer satisfaction."
    )
}

# Frases de despedida
FAREWELL_PHRASES = [
    "goodbye",
    "bye",
    "see you",
    "thank you very much",
    "that's all i needed"
]

# Mapeo de quick intents
QUICK_INTENTS = {
    "hi my name is alex can you help me": "Hello Alex! Absolutely, I'm here to help. What can I do for you today?",
    "what is your name": "I am Jessica, your AI assistant from NetConnect Services.",
    "Great Jessica, thank you Right now we are going to make a Jesy.Ai demo I will call you back in 5 minutes": "Perfect! I will be here and ready to assist with the demo. Talk to you soon!",
    "are you a real person": "I'm a virtual AI assistant, but I'll do my best to assist you like a real agent.",
    "Hello Jessica I am here recording you can you please tell me what makes Jesy.Ai so special": "Hello! Thank you for asking. Jesy.Ai is special because it provides intelligent, natural-sounding customer support 24/7, adapts to virtually any industry’s needs, and can handle a high volume of inquiries efficiently. Our goal is to make sure every customer interaction is fast, smooth, and personalized, no matter the business.",
    "my internet is running slow": "slow_internet",
    "when is my payment due and how much do i owe": "payment_due",
    "what are your support hours": "support_hours",
    "i think thats all i needed thanks": "issue_resolved",
    "in what businesses can we implement jesyai": "implementation"
}