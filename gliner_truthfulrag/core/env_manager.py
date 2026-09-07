from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    # only cuda is supported
    MAP_LOCATION: str = "cuda:0"
    USE_FLASHDEBERTA: str = "1"
    PROMPT_DIR: str = "gliner_truthfulrag/prompts"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
