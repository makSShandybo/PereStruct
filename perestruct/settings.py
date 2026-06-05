from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="perestruct/.env", env_file_encoding="utf-8", extra="ignore")

    # Yandex
    ocr_api_key: str = Field(default="", description="API KEY for Yandex Cloud")
    llm_api_key: str = Field(default="", description="API KEY for Yandex Cloud")
    folder_id: str = Field(default="", description="Folder ID for Yandex Clound")
    self_token: str = Field(default="", description="Self Token for Yandex Clound")




@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
