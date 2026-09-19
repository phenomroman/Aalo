from functools import lru_cache
from typing import List
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    ollama_base_url: str = "http://127.0.0.1:11434"
    api_key: str = ""
    allowed_origins: str = "*"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        clean = v.strip()
        if not clean or clean == "aalo-secret-key-change-me":
            raise ValueError(
                "A secure API_KEY must be set in your .env file before starting the gateway."
            )
        return clean

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

