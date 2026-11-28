"""Application configuration and secret handling."""
from functools import lru_cache
import os
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("RAMI_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "jervis-backend"
    service_version: str = "1.0.0"
    app_env: str = "development"
    enable_docs: bool | None = None
    log_level: str = "INFO"
    cors_allowed_origins: str = ""

    # Sensitive values are excluded from repr/str so secrets never leak in logs.
    service_token: str | None = Field(default=None, repr=False)
    database_url: str | None = Field(default=None, repr=False)

    r2_account_id: str | None = None
    r2_access_key_id: str | None = Field(default=None, repr=False)
    r2_secret_access_key: str | None = Field(default=None, repr=False)
    r2_bucket_name: str | None = None
    r2_endpoint_url: str | None = None

    @field_validator(
        "service_token",
        "database_url",
        "r2_account_id",
        "r2_access_key_id",
        "r2_secret_access_key",
        "r2_bucket_name",
        "r2_endpoint_url",
        mode="before",
    )
    @classmethod
    def _empty_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned if cleaned else None

    def __str__(self) -> str:
        return repr(self)

    @property
    def docs_enabled(self) -> bool:
        if self.enable_docs is not None:
            return self.enable_docs
        return self.app_env.strip().lower() != "production"

    @property
    def cors_origins(self) -> list[str]:
        origins: list[str] = []
        for raw in self.cors_allowed_origins.split(","):
            cleaned = raw.strip().rstrip("/")
            if cleaned and cleaned != "*":
                origins.append(cleaned)
        return origins

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url)

    @property
    def r2_endpoint(self) -> str | None:
        if self.r2_endpoint_url:
            return self.r2_endpoint_url.rstrip("/")
        if self.r2_account_id:
            return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"
        return None

    @property
    def r2_configured(self) -> bool:
        return bool(
            self.r2_endpoint
            and self.r2_access_key_id
            and self.r2_secret_access_key
            and self.r2_bucket_name
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
