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
logger.info("Environment variables loaded")

# Initialize Flask app
app = Flask(__name__)
logger.info("Flask app initialized")

# Configure CORS for Dataiku CodeStudio
CORS(app,
    origins=["*"],  # Simplified for development
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    supports_credentials=True)

logger.info("CORS configured")

# Add request logging middleware
@app.before_request
def log_request_info():
    logger.debug('Headers: %s', dict(request.headers))
    logger.debug('Path: %s', request.path)
    logger.debug('Method: %s', request.method)
    logger.debug('Args: %s', dict(request.args))
    logger.debug('Origin: %s', request.headers.get('Origin', 'No origin'))

# Add a health check route
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Flask app is running"})

# Add a simple test route
@app.route('/api/test', methods=['GET'])
def test_route():
    return jsonify({"message": "Flask API is working!", "status": "success"})

# Register the fetch_api blueprint with /api prefix
try:
    app.register_blueprint(fetch_api, url_prefix='/api')
    logger.info("fetch_api blueprint registered successfully with /api prefix")
except Exception as e:
    logger.error(f"Blueprint registration failed: {e}")
    raise

# Debug: Print blueprint info
logger.info(f"fetch_api blueprint name: {fetch_api.name}")
logger.info(f"fetch_api blueprint url_prefix: {getattr(fetch_api, 'url_prefix', 'None')}")

if __name__ == "__main__":
    logger.info("=== ENTERING MAIN ===")
    
    # Log all registered routes
    logger.info("\n==== REGISTERED ROUTES ====")
    for rule in sorted(app.url_map.iter_rules(), key=lambda x: str(x)):
        logger.info(f"Route: {rule.rule}, Methods: {rule.methods}, Endpoint: {rule.endpoint}")
    logger.info("===========================\n")
    
    # Run the app
    port = int(os.getenv("VITE_API_PORT", 5000))
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    
    logger.info(f"Starting Flask app on {host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
    logger.info("=== EXITING MAIN ===")