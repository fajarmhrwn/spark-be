from flask import Flask
from fetch_api import fetch_api
from dotenv import load_dotenv
from flask_cors import CORS
import os

load_dotenv()

from webaiku.extension import WEBAIKU


app = Flask(__name__)
# Fix: Use 'origins' parameter correctly
CORS(app,
     origins=["http://localhost:4200", "http://localhost:5173", "http://localhost:3000"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=True)
WEBAIKU(
    app, "webapps/spark", int(os.getenv("VITE_API_PORT"))
)
WEBAIKU.extend(app, [fetch_api])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("VITE_API_PORT")), debug=True)
