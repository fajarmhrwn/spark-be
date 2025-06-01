from flask import Flask
from fetch_api import fetch_api
from dotenv import load_dotenv
from flask_cors import CORS
import os

load_dotenv()
from webaiku.extension import WEBAIKU

app = Flask(__name__)

# Fix: Configure CORS properly - remove the "*" as it conflicts with specific origins
CORS(app,
     origins=["http://localhost:4200", "http://localhost:5300", "http://localhost:3000","*","**"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)  # Add this if you need to send cookies/auth

# Alternative: If you want to allow all origins during development (less secure)
# CORS(app, origins="*")

WEBAIKU(
    app, "webapps/spark", int(os.getenv("VITE_API_PORT"))
)

WEBAIKU.extend(app, [fetch_api])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("VITE_API_PORT")), debug=True)
