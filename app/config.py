from functools import lru_cache
from typing import List
from pydantic import model_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    ollama_base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    api_keys: str = ""
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    system_prompt: str = Field(
        default="You are Alo, a highly capable and intelligent AI assistant powering the Aalo application. You were created by Masrur Ahmed Roman at phenomroman.com", 
        description="Core personality instructions for the LLM"
    )

    @property
    def valid_api_keys(self) -> List[str]:
        """Return list of distinct, sanitized API keys from API_KEY and/or API_KEYS."""
        combined = f"{self.api_key},{self.api_keys}"
        keys: List[str] = []
        for raw in combined.split(","):
            clean = raw.strip()
            if clean and clean != "aalo-secret-key-change-me" and clean not in keys:
                keys.append(clean)
        return keys

    @model_validator(mode="after")
    def validate_keys(self) -> "Settings":
        if not self.valid_api_keys:
            raise ValueError(
                "No valid API keys configured. Set API_KEY or API_KEYS (comma-separated) in your .env file."
            )
        return self

    @property
    def cors_origins(self) -> List[str]:
        raw = self.allowed_origins.strip()
        if not raw or raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def cors_allow_credentials(self) -> bool:
        # Browser security specifications forbid allow_credentials=True when allow_origins=["*"]
        return self.cors_origins != ["*"]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
