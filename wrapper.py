import json
import logging
import requests

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
        response.raise_for_status()
        logger.info("Chat in Workspace Executed: %s", response.json())
        return response.json()


if __name__ == "__main__":
    # Example usage
    BASE_URL = "http://localhost:3001/api/v1"
    API_KEY = "14QNGK1-525M7JF-GWZ4CE7-VH13DBT"  #

    api = APIWrapper(base_url=BASE_URL, api_key=API_KEY)

    try:
        # Create a workspace
        workspace = api.create_workspace(
            workspace_name="testowe",
            similarity_threshold=0.7,
            openai_temp=0.7,
            openai_history=20,
            openai_prompt="Custom prompt for responses",
            query_refusal_response="Custom refusal message",
            chat_mode="chat",
            top_n=4
        )

        # Get workspace details
        workspace_details = api.get_workspace_details("testowe")

        # Get available models
        models = api.get_available_models()

        # Chat within a specific workspace
        workspace_chat_response = api.chat_in_workspace(
            slug="testowe",
            message="Describe me the Merito University?",
            mode="chat",
            session_id="identifier-to-partition-chats-by-external-id"
        )
        logger.info("Workspace Chat Response: %s", workspace_chat_response)

        # Delete workspace
        if api.delete_workspace('testowe'):
            logger.info("Workspace Deleted Successfully")

    except Exception as e:
        print("An error occurred:", str(e))
