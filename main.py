import os
import weaviate

from dotenv import load_dotenv
from pathlib import Path

from flask import Flask, request, jsonify
from pydantic import ValidationError
from weaviate.classes.init import Auth

from Tools_Agents.product import product_call
from Tools_Agents.domain import domain_call
from Tools_Agents.product_ask_demo import product_ask_demo_call
from Tools_Agents.domain_gp_demo import domain_gp_demo_call
from Tools_Agents.domain_ask_demo import domain_ask_demo_call
from Response_Agents.large_context_agent import large_context_call
from Response_Agents.no_context_agent import no_context_call

# Load environment variables
script_dir = Path(__file__).resolve().parent
env_path = script_dir.parent / '.env'
load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEAVIATE_CLIENT_ID = os.environ.get("WEAVIATE_CLIENT_ID")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")

# Initialize Weaviate client
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_CLIENT_ID,  # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),  # Replace with your Weaviate Cloud key
    headers={'X-OpenAI-Api-key': OPENAI_API_KEY}  # Replace with your OpenAI API key
)

# demo_client = weaviate.connect_to_custom(
#     http_host="52.220.0.151",
#     http_port=8080,
#     http_secure=False,
#     grpc_host="52.220.0.151",
#     grpc_port=50051,
#     grpc_secure=False,
#     skip_init_checks=True,
#     headers={'X-OpenAI-Api-key': OPENAI_API_KEY, 'X-API-KEY': '4950cef6afb7448b8bf5bd9e8355e1d8'}
# )

# Initialize Flask app
app = Flask(__name__)

@app.route('/chat/product', methods=['POST'])
def post_product():
    try:
        data = request.get_json()
        response = product_call(data)
        return response
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
@app.route('/chat/domain', methods=['POST'])
def post_domain():
    try:
        data = request.get_json()
        response = domain_call(data)
        return response
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
@app.route('/chat/product_ask_demo', methods=['POST'])
def post_product_ask_demo():
    try:
        data = request.get_json()
        response = product_ask_demo_call(data)
        return response
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
@app.route('/chat/domain_gp_demo', methods=['POST'])
def post_domain_gp_demo():
    try:
        data = request.get_json()
        response = domain_gp_demo_call(data)
        return response
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
@app.route('/chat/domain_ask_demo', methods=['POST'])
def post_domain_ask_demo():
    try:
        data = request.get_json()
        response = domain_ask_demo_call(data)
        return response
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
@app.route('/chat/large_context', methods=['POST'])
def post_large_context():
    try:
        data = request.get_json()
        response = large_context_call(data)
        return response
    
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500
    
@app.route('/chat/no_context', methods=['POST'])
def post_no_context():
    try:
        data = request.get_json()
        response = no_context_call(data)
        return response
    
    except ValidationError as e:
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        return jsonify({"error": "An error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', debug=True)
    finally:
        client.close()
        #demo_client.close()
        print("Weaviate client connection closed.")