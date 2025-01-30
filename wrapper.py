import json
import logging
import os
import requests
import time
from datetime import datetime
from helpers import txt_to_pdf
from metrics import bleu
from metrics import rouge


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIWrapper:
    def __init__(self, base_url, api_key):
        """
        Initialize the API Wrapper and authenticate the user.

        :param base_url: Base URL for the API.
        :param api_key: API key for authentication.
        """
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "accept": "application/json"
        }

        # Perform authentication
        auth_url = f"{self.base_url}/auth"
        response = requests.get(auth_url, headers=self.headers)

        if response.status_code == 200:
            logger.info("Authentication Successful: %s", response.json())
        else:
            logger.error("Authentication Failed: %s, %s", response.status_code, response.text)
            raise Exception(f"Authentication Failed: {response.status_code}, {response.text}")

    def create_workspace(self, workspace_name, similarity_threshold=0.7, openai_temp=0.7,
                         openai_history=20, openai_prompt="Custom prompt for responses",
                         query_refusal_response="Custom refusal message", chat_mode="chat",
                         top_n=4):
        """
        Create a new workspace.

        :param workspace_name: Name of the workspace to create.
        :param similarity_threshold: Similarity threshold for queries.
        :param openai_temp: Temperature for OpenAI responses.
        :param openai_history: Number of OpenAI history messages.
        :param openai_prompt: Custom prompt for OpenAI responses.
        :param query_refusal_response: Custom refusal message for invalid queries.
        :param chat_mode: Mode of chat operation.
        :param top_n: Number of top results to retrieve.
        """
        logger.info("Creating Workspace")
        endpoint = f"{self.base_url}/workspace/new"
        payload = json.dumps({
            "name": workspace_name,
            "similarityThreshold": similarity_threshold,
            "openAiTemp": openai_temp,
            "openAiHistory": openai_history,
            "openAiPrompt": openai_prompt,
            "queryRefusalResponse": query_refusal_response,
            "chatMode": chat_mode,
            "topN": top_n
        })
        response = requests.post(endpoint,
                                 headers={**self.headers, "Content-Type": "application/json"},
                                 data=payload)
        response.raise_for_status()
        logger.info("Status Code %s %s", response.status_code, 'OK')
        logger.info("Workspace Created: %s", response.json())
        return response.json()

    def delete_workspace(self, workspace_slug):
        """
        Delete a workspace.

        :param workspace_slug: ID of the workspace to delete.
        """
        logger.info("Deleting Workspace")
        endpoint = f"{self.base_url}/workspace/{workspace_slug}"
        payload = ""
        response = requests.delete(endpoint, headers=self.headers, data=payload)
        if response.status_code == 200:
            logger.info("Status Code %s %s", response.status_code, response.text)
            return True
        else:
            logger.error("Failed to Delete Workspace: %s, %s", workspace_slug, response.text)
            response.raise_for_status()

    def get_workspace_details(self, slug):
        """
        Get details of a specific workspace.
        :param slug: slug of the workspace.
        """
        endpoint = f"{self.base_url}/workspace/{slug}"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        logger.info("Workspace Details: %s", response.json())
        return response.json()

    def get_available_models(self):
        """
        Get the list of available models with detailed information.
        """
        endpoint = f"{self.base_url}/openai/models"
        response = requests.get(endpoint, headers=self.headers)
        response.raise_for_status()
        models_data = response.json()

        # Check if response contains 'data' key
        if "data" in models_data and isinstance(models_data["data"], list):
            models = []
            for model in models_data["data"]:
                models.append({
                    "name": model.get("name", ""),
                    "model": model.get("model", ""),
                    "llm": {
                        "provider": model.get("llm", {}).get("provider", ""),
                        "model": model.get("llm", {}).get("model", "")
                    }
                })
            logger.info("Available Models: %s", models)
            return models
        else:
            logger.error("Unexpected response structure: %s", models_data)
            raise ValueError("Unexpected response structure.")

    def update_model(self, workspace_slug, chat_provider, chat_model):
        """
        Update the chat model of a workspace.

        :param workspace_slug: The identifier of the workspace to update.
        :param chat_provider: The chat provider to set.
        :param chat_model: The chat model to use.
        """
        logger.info("Updating model for workspace: %s", workspace_slug)
        endpoint = f"{self.base_url}/workspace/{workspace_slug}/update"
        payload = json.dumps({
            "chatProvider": chat_provider,
            "chatModel": chat_model
        })
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.post(endpoint, headers=headers, data=payload)

        if response.status_code == 200:
            logger.info("Model Updated Successfully: %s", response.json())
            return response.json()
        else:
            logger.error("Failed to Update Model: %s, %s", response.status_code, response.text)
            response.raise_for_status()

    def upload_document(self, file_path):
        """
        Upload a document to the API.

        :param file_path: Path to the file to be uploaded.
        """
        logger.info("Uploading document: %s", file_path)
        endpoint = f"{self.base_url}/document/upload"
        files = {
            'file': (file_path.split('/')[-1], open(file_path, 'rb'), 'application/pdf')
        }
        headers = {**self.headers}

        response = requests.post(endpoint, headers=headers, files=files)

        if response.status_code == 200:
            logger.info("Document Uploaded Successfully: %s", response.json())
            location = response.json().get("documents", [{}])[0].get("location", "Unknown")
            logger.info("Document Uploaded Successfully. Location: %s", location)
            return location
        else:
            logger.error("Failed to Upload Document: %s, %s", response.status_code, response.text)
            response.raise_for_status()

    def embed_document_to_workspace(self, workspace_slug, location):
        """
        Embed a document into a workspace.

        :param workspace_slug: The identifier of the workspace.
        :param location: The document location to embed.
        """
        logger.info("Embedding document to workspace: %s", workspace_slug)
        endpoint = f"{self.base_url}/workspace/{workspace_slug}/update-embeddings"
        payload = json.dumps({
            "adds": [location],
            "deletes": []
        })
        headers = {**self.headers, "Content-Type": "application/json"}

        response = requests.post(endpoint, headers=headers, data=payload)

        if response.status_code == 200:
            logger.info("Document Embedded Successfully: %s", response.json())
            return response.json()
        else:
            logger.error("Failed to Embed Document: %s, %s", response.status_code, response.text)
            response.raise_for_status()

    def chat_in_workspace(self, slug, message, mode="chat", session_id=None):
        """
        Chat within a specific workspace.

        :param slug: slug of the workspace to chat in.
        :param message: The message to send.
        :param mode: Mode of the chat (e.g., "chat").
        :param session_id: Optional session identifier for partitioning chats.
        """
        endpoint = f"{self.base_url}/workspace/{slug}/chat"
        payload = {
            "message": message,
            "mode": mode
        }
        if session_id:
            payload["sessionId"] = session_id

        response = requests.post(endpoint,
                                 headers={**self.headers, "Content-Type": "application/json"},
                                 json=payload)
        time.sleep(360)
        response.raise_for_status()
        logger.info("Chat in Workspace Executed: %s", response.json())
        text_response = response.json()["textResponse"]
        return text_response

if __name__ == "__main__":
    # Example usage
    BASE_URL = "http://localhost:3001/api/v1"
    API_KEY = "2WDTDG6-A7M4T20-QCSC2Q0-CPZ14M5"
    WORKSPACE_SLUG = "masterdegree"

    api = APIWrapper(base_url=BASE_URL, api_key=API_KEY)

    try:
        # Create a workspace
        workspace = api.create_workspace(
            workspace_name=WORKSPACE_SLUG,
            similarity_threshold=0.7,
            openai_temp=0.7,
            openai_history=20,
            openai_prompt="Custom prompt for responses",
            query_refusal_response="Custom refusal message",
            chat_mode="chat",
            top_n=4
        )

        # Get workspace details
        workspace_details = api.get_workspace_details(WORKSPACE_SLUG)

        # Get available models
        models = api.get_available_models()

        # Update Model
        # update_response = api.update_model(
        #     workspace_slug=WORKSPACE_SLUG,
        #     chat_provider="anythingllm_ollama",
        #     chat_model="Llama3.2 3B:latest"
        # )
        # logger.info("Update Model Response: %s", update_response)

        # Upload Document
        document_location = api.upload_document("files/reference.pdf")
        logger.info("Upload Document Location: %s", document_location)

        # Embed file to workspace
        embed_response = api.embed_document_to_workspace(WORKSPACE_SLUG, document_location)
        logger.info("Embed Document Response: %s", embed_response)

        # Chat within a specific workspace
        workspace_chat_response = api.chat_in_workspace(
            slug=WORKSPACE_SLUG,
            message="Explain me for loop in Python?",
            mode="chat",
            session_id="master-session"
        )
        logger.info("Returned answer: %s", workspace_chat_response)

        # Collect Chat Data to file
        # Save the text to a file
        txt_filename = "txt_output/text_response.txt"

        logger.info("Saving answer to text file")

        with open(txt_filename, "w", encoding="utf-8") as file:
            file.write(workspace_chat_response)

        logger.info(f"Text file '{txt_filename}' saved successfully.")

        # Create and Save PDF
        logger.info("Saving answer to pdf file")
        pdf_filename = "text_response.pdf"
        pdf_file_path = os.path.join('pdf_output', f'{pdf_filename}')
        txt_to_pdf(os.path.join(txt_filename), pdf_file_path)

        logger.info(f"PDF file '{pdf_file_path}' saved successfully.")

    except Exception as e:
        logger.error("An error occurred:", str(e))

    # Delete workspace
    if api.delete_workspace(WORKSPACE_SLUG):
        logger.info("Workspace Deleted Successfully")

    # Generate timestamp and output file name
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # Running Calculations
    logger.info('Running BLEU calculation')
    bleu.bleu_calculation(pdf_file_path)

    logging.info('Running ROUGE calculation')
    rouge.rouge_calculation(pdf_file_path)
