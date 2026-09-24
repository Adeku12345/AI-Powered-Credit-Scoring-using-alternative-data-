from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from backend.core.config import settings
from backend.core.logging import setup_logger

logger = setup_logger("llm-config")

def get_llm(temperature: float = None, model: str = None) -> BaseChatModel | None:
    temp = temperature or settings.LLM_TEMPERATURE
    model_name = model or settings.LLM_MODEL
    if not settings.OPENAI_API_KEY:
        logger.warning("⚠️ OPENAI_API_KEY not set — using template explanations")
        return None
    return ChatOpenAI(
        model=model_name, temperature=temp, api_key=settings.OPENAI_API_KEY,
        timeout=30, max_retries=2
    )

llm = get_llm()