# Import necessary libraries (modules)
import argparse
import json
import logging
import os
import requests
import time
import configparser
from datetime import datetime
from helpers.helpers import txt_to_pdf
from metrics import bleu, rouge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description="API Wrapper Script with Config File Parameter")
parser.add_argument("--config", type=str, required=True, help="Path to the configuration file (config.ini)")
args = parser.parse_args()

# Load configuration from the specified file
config = configparser.ConfigParser()
config.read(args.config)

BASE_URL = config.get('API', 'BASE_URL')
API_KEY = config.get('API', 'API_KEY')
WORKSPACE_SLUG = config.get('API', 'WORKSPACE_SLUG')
CHAT_PROVIDER = config.get('MODEL', 'CHAT_PROVIDER')
CHAT_MODEL = config.get('MODEL', 'CHAT_MODEL')
MODEL_DOWNLOADED = config.getboolean('MODEL', 'MODEL_DOWNLOADED')  # Convert to boolean
UPLOAD_FILE = config.getboolean('MODEL', 'UPLOAD_FILE')  # Convert to boolean
REFERENCE_FILE = config.get('MODEL', 'REFERENCE_FILE')
SIMILARITY_THRESHOLD = config.get('SETTINGS', 'SIMILARITY_THRESHOLD')
OPEN_AI_TEMP = config.get('SETTINGS', 'OPEN_AI_TEMP')
OPEN_AI_HISTORY = config.get('SETTINGS', 'OPEN_AI_HISTORY')
QUESTION_TO_CHAT = config.get('PROMPT', 'QUESTION_TO_CHAT')

# Wrapper Class
class APIWrapper:
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json"
        }

        auth_url = f"{self.base_url}/auth"
        response = requests.get(auth_url, headers=self.headers)
        if response.status_code == 200:
            logger.info(f"Authentication Successful: {response.json()}")
        else:
            logger.error(f"Authentication Failed: {response.status_code}, {response.text}")
            raise Exception(f"Authentication Failed: {response.status_code}, {response.text}")

    def create_workspace(self, workspace_name, **kwargs):
        logger.info("Creating Workspace")
        endpoint = f"{self.base_url}/workspace/new"
        payload = json.dumps({"name": workspace_name, **kwargs})
        response = requests.post(endpoint, headers={**self.headers, "Content-Type": "application/json"}, data=payload)
        response.raise_for_status()
        logger.info(f"Workspace Created: {response.json()}")
        return response.json()

    def update_model(self, workspace_slug, chat_provider, chat_model):
        logger.info(f"Updating model for workspace: {workspace_slug}")
        endpoint = f"{self.base_url}/workspace/{workspace_slug}/update"
        payload = json.dumps({"chatProvider": chat_provider, "chatModel": chat_model})
        response = requests.post(endpoint, headers={**self.headers, "Content-Type": "application/json"}, data=payload)
        response.raise_for_status()
        logger.info(f"Model Updated Successfully: {response.json()}")
        return response.json()

    def upload_document(self, file_path):
        logger.info(f"Uploading document: {file_path}")
        endpoint = f"{self.base_url}/document/upload"
        with open(file_path, 'rb') as file:
            files = {'file': (os.path.basename(file_path), file, 'application/pdf')}
            response = requests.post(endpoint, headers=self.headers, files=files)
            response.raise_for_status()
            location = response.json().get("documents", [{}])[0].get("location", "Unknown")
            logger.info(f"Document Uploaded. Location: {location}")
            return location

    def embed_document_to_workspace(self, workspace_slug, location):
        logger.info(f"Embedding document to workspace: {workspace_slug}")
        endpoint = f"{self.base_url}/workspace/{workspace_slug}/update-embeddings"
        payload = json.dumps({"adds": [location], "deletes": []})
        response = requests.post(endpoint, headers={**self.headers, "Content-Type": "application/json"}, data=payload)
        response.raise_for_status()
        logger.info(f"Document Embedded Successfully: {response.json()}")
        return response.json()

    def chat_in_workspace(self, slug, message, mode="chat", session_id=None):
        endpoint = f"{self.base_url}/workspace/{slug}/chat"
        payload = {"message": message, "mode": mode, "sessionId": session_id} if session_id else {"message": message, "mode": mode}
        response = requests.post(endpoint, headers={**self.headers, "Content-Type": "application/json"}, json=payload)
        time.sleep(2)  # Reduce sleep time for responsiveness
        response.raise_for_status()
        logger.info(f"Chat Response: {response.json()}")
        return response.json()["textResponse"]

    def delete_workspace(self, workspace_slug):
        logger.info(f"Deleting Workspace: {workspace_slug}")
        endpoint = f"{self.base_url}/workspace/{workspace_slug}"
        response = requests.delete(endpoint, headers=self.headers)
        response.raise_for_status()
        logger.info(f"Workspace Deleted: {workspace_slug}")
        return response.status_code

    def remove_documents(self):
        logger.info("Removing Documents")
        endpoint = f"{self.base_url}/system/remove-documents"
        payload = json.dumps({"names": "custom-documents"})
        response = requests.delete(endpoint, headers={**self.headers, "Content-Type": "application/json"}, data=payload)
        response.raise_for_status()
        logger.info(f"Documents Removed")
        return response.json()

# Main function
if __name__ == "__main__":
    api = APIWrapper(base_url=BASE_URL, api_key=API_KEY)
    txt_filename = "output/txt_output/text_response.txt"
    pdf_filename = "output/pdf_output/text_response.pdf"
    reference_file = REFERENCE_FILE

    try:
        workspace = api.create_workspace(
            workspace_name=WORKSPACE_SLUG,
            similarityThreshold=SIMILARITY_THRESHOLD,
            openAiTemp=OPEN_AI_TEMP,
            openAiHistory=OPEN_AI_HISTORY,
            openAiPrompt="Custom prompt for responses",
            queryRefusalResponse="Custom refusal message",
            chatMode="chat",
            topN=4
        )

        if not MODEL_DOWNLOADED:
            logger.info(f"Download and Load Model {CHAT_MODEL}")
            api.update_model(WORKSPACE_SLUG, CHAT_PROVIDER, CHAT_MODEL)
        
        api.update_model(WORKSPACE_SLUG, CHAT_PROVIDER, CHAT_MODEL)

        if UPLOAD_FILE:
            document_location = api.upload_document(reference_file)
            api.embed_document_to_workspace(WORKSPACE_SLUG, document_location)

        response_text = api.chat_in_workspace(WORKSPACE_SLUG, QUESTION_TO_CHAT, "chat", "master-session")
        logger.info(f"Chat Answer: {response_text}")

        with open(txt_filename, "w", encoding="utf-8") as file:
            file.write(response_text)

        logger.info(f"Text file saved: {txt_filename}")
        txt_to_pdf(txt_filename, pdf_filename)

        logger.info(f"PDF file saved: {pdf_filename}")
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        bleu.bleu_calculation(pdf_filename, reference_file)
        rouge.rouge_calculation(pdf_filename, reference_file)
        
        logger.info(f"Delete workspace")
        api.delete_workspace(WORKSPACE_SLUG)

        logger.info(f"Remove files from system")
        api.remove_documents()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
