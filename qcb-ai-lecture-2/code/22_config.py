from dataclasses import dataclass
import os
from openai import OpenAI
@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "local"
    model: str = "gpt-oss-20b"
    temperature: float = 1.0
    max_tokens: int = 2000
    
    @classmethod
    def from_env(cls):
        return cls(base_url=os.getenv("LLM_BASE_URL", cls.base_url),
                   model=os.getenv("LLM_MODEL", cls.model)
        )

config = LLMConfig.from_env()
client = OpenAI(base_url=config.base_url, api_key=config.api_key)
