import json
import logging
import os
import requests
import time
import configparser
from datetime import datetime
from helpers import txt_to_pdf
from metrics import bleu, rouge

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

BASE_URL = config.get('API', 'BASE_URL')
API_KEY = config.get('API', 'API_KEY')
WORKSPACE_SLUG = config.get('API', 'WORKSPACE_SLUG')
CHAT_PROVIDER = config.get('MODEL', 'CHAT_PROVIDER')
CHAT_MODEL = config.get('MODEL', 'CHAT_MODEL')
MODEL_LOADED = config.get('MODEL', 'MODEL_LOADED')
SIMILARITY_THRESHOLD = config.get('SETTINGS', 'SIMILARITY_THRESHOLD')
OPEN_AI_TEMP = config.get('SETTINGS', 'OPEN_AI_TEMP')
OPEN_AI_HISTORY = config.get('SETTINGS', 'OPEN_AI_HISTORY')
QUESTION_TO_CHAT = config.get('PROMPT', 'QUESTION_TO_CHAT')


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
        logger.info(f"Model Updated Successfully: {response_json()}")
        return response.json()

    def upload_document(self, file_path):
        logger.info(f"Uploading document: {file_path}")
        endpoint = f"{self.base_url}/document/upload"
        files = {'file': (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')}
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
        logger.info(f"Document Embedded Successfully:{response.json()}")
        return response.json()

    def chat_in_workspace(self, slug, message, mode="chat", session_id=None):
        endpoint = f"{self.base_url}/workspace/{slug}/chat"
        payload = {"message": message, "mode": mode, "sessionId": session_id} if session_id else {"message": message, "mode": mode}
        response = requests.post(endpoint, headers={**self.headers, "Content-Type": "application/json"}, json=payload)
        time.sleep(720)
        response.raise_for_status()
        logger.info(f"Chat Response: {response.json()}", )
        return response.json()["textResponse"]

if __name__ == "__main__":
    api = APIWrapper(base_url=BASE_URL, api_key=API_KEY)
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

        if MODEL_LOADED == True:
            logger.info(f"Download and Load Model {CHAT_MODEL}")
            api.update_model(WORKSPACE_SLUG, CHAT_PROVIDER, CHAT_MODEL)

        document_location = api.upload_document("files/reference.pdf")
        api.embed_document_to_workspace(WORKSPACE_SLUG, document_location)

        response_text = api.chat_in_workspace(WORKSPACE_SLUG, QUESTION_TO_CHAT, "chat", "master-session")
        logger.info(f"Chat Answer: {response_text}")

        txt_filename = "txt_output/text_response.txt"
        with open(txt_filename, "w", encoding="utf-8") as file:
            file.write(response_text)
        logger.info(f"Text file saved: {txt_filename}")

        pdf_filename = "pdf_output/text_response.pdf"
        txt_filename = "txt_output/text_response.txt"
        txt_to_pdf(txt_filename, pdf_filename)
        logger.info(f"PDF file saved: {pdf_filename}")

        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        bleu.bleu_calculation(pdf_filename)
        rouge.rouge_calculation(pdf_filename)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
