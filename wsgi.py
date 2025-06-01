from flask import Flask, request, jsonify
from fetch_api import fetch_api
from dotenv import load_dotenv
from flask_cors import CORS
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import WEBAIKU after environment is loaded
from webaiku.extension import WEBAIKU

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for Dataiku CodeStudio
CORS(app, 
     # Allow more origins for CodeStudio environment
     origins=["*", 
              "http://localhost:4200", 
              "http://localhost:5173", 
              "http://localhost:3000",
              # Add Dataiku CodeStudio domains if they have specific patterns
              "https://*.dataiku.com"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     supports_credentials=True)

# Add request logging middleware
@app.before_request
def log_request_info():
    logger.debug('Headers: %s', dict(request.headers))
    logger.debug('Path: %s', request.path)
    logger.debug('Method: %s', request.method)
    logger.debug('Args: %s', dict(request.args))
    logger.debug('Origin: %s', request.headers.get('Origin', 'No origin'))

# Initialize WEBAIKU with your app - ensure paths are correct for CodeStudio
webaiku_instance = WEBAIKU(
    app, 
    "webapps/spark",  # Check if this path is correct in CodeStudio
    int(os.getenv("VITE_API_PORT", 5000))  # Provide default if env var is missing
)

# Register the blueprint through WEBAIKU
webaiku_instance.extend(app, [fetch_api])

# Print the app's config for debugging
logger.debug("Flask App Config: %s", {k: v for k, v in app.config.items() if not k.startswith('_')})

if __name__ == "__main__":
    # Log all registered routes
    logger.info("\n==== REGISTERED ROUTES ====")
    for rule in sorted(app.url_map.iter_rules(), key=lambda x: str(x)):
        logger.info(f"Route: {rule}, Methods: {rule.methods}")
    logger.info("===========================\n")
    
    # Run the app with CodeStudio-friendly settings
    port = int(os.getenv("VITE_API_PORT", 5000))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)  # Use 0.0.0.0 to accept connections from anywhere
