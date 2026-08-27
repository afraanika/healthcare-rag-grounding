from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b-instruct"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "data/chroma"
    chunk_size: int = 650
    chunk_overlap: int = 100


settings = Settings()
