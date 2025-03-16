import time
import logging
from abc import ABC, abstractmethod
from openai import OpenAI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, messages, max_tokens=50, temperature=0.8, top_p=0.95, n=1):
        """Generate completions from the model."""
        pass

class OpenAICompatibleClient(ModelClient):
    """Client for interacting with OpenAI API compatible endpoints (including Ollama)."""
    
    def __init__(self, base_url="http://localhost:11434/v1", model="llama2"):
        self.model = model
        
        # Initialize OpenAI client with the base URL
        try:
            self.client = OpenAI(
                base_url=base_url,
                api_key="ollama",  # Ollama doesn't require a real API key but the client needs something
            )
            logger.info(f"Successfully connected to API endpoint. Using model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client with base URL {base_url}: {e}")
            raise
    
    def generate(self, messages, max_tokens=50, temperature=0.8, top_p=0.95, n=1):
        """Generate completions using OpenAI-compatible API."""
        responses = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    n=n
                )
                
                # Extract message content from all choices
                for choice in completion.choices:
                    content = choice.message.content
                    responses.append(content)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                logger.error(f"Error in attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retrying
                else:
                    # On last attempt, add empty responses to match expected count
                    responses.extend([""] * (n - len(responses)))
        
        return responses

class MockClient(ModelClient):
    """Mock client for testing without API calls."""
    
    def generate(self, messages, max_tokens=50, temperature=0.8, top_p=0.95, n=1):
        """Return mock responses for testing."""
        # Extract the code diff from the last user message
        last_user_content = next((msg["content"] for msg in reversed(messages) 
                                if msg["role"] == "user"), "")
        
        # Create a simple mock response based on the content
        mock_responses = []
        for i in range(n):
            # Simple heuristic to generate different mock commit messages
            if "add" in last_user_content.lower():
                mock_responses.append(f"Add new functionality to the codebase (mock {i+1})")
            elif "fix" in last_user_content.lower():
                mock_responses.append(f"Fix issue in the implementation (mock {i+1})")
            elif "remove" in last_user_content.lower():
                mock_responses.append(f"Remove deprecated code (mock {i+1})")
            else:
                mock_responses.append(f"Update code implementation (mock {i+1})")
        
        return mock_responses

def get_client(client_type, model="llama2", base_url="http://localhost:11434/v1"):
    """Factory function to get the appropriate client."""
    if client_type.lower() == "openai":
        return OpenAICompatibleClient(base_url=base_url, model=model)
    elif client_type.lower() == "mock":
        return MockClient()
    else:
        raise ValueError(f"Unsupported client type: {client_type}")