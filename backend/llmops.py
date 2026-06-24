import os
from typing import Dict, Any
from langfuse import observe, Langfuse
from nemoguardrails import LLMRails, RailsConfig

# Initialize Langfuse Client to validate configuration
langfuse_client = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

def init_observability() -> None:
    """Validates the Langfuse connection upon startup."""
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        print("Warning: Langfuse credentials missing. Observability is disabled.")
        return
        
    if not langfuse_client.auth_check():
        print("Warning: Langfuse authentication failed. Check API keys.")
    else:
        print("Langfuse observability initialized successfully.")

def init_guardrails() -> LLMRails:
    """Initializes the NeMo Guardrails application from the configuration directory."""
    config_path = os.path.join(os.path.dirname(__file__), "guardrails_config")
    config = RailsConfig.from_path(config_path)
    return LLMRails(config)

# Singleton guardrail instance
rails_app = None

@observe(as_type="generation")
async def safe_generate(query: str, context: str = "") -> str:
    """
    Applies NeMo Guardrails to the outgoing generation request.
    This function evaluates the query against predefined safety and topical rails,
    and logs the trace to Langfuse automatically via the @observe decorator.
    """
    global rails_app
    if rails_app is None:
        rails_app = init_guardrails()
        
    messages = [
        {"role": "context", "content": {"relevant_chunks": context}},
        {"role": "user", "content": query}
    ]
    
    # Generate response securely through NeMo Guardrails
    response = await rails_app.generate_async(messages=messages)
    
    if isinstance(response, dict):
        return response.get("content", "Error generating response.")
    return str(response)
