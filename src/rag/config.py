from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API keys
    openai_api_key: str
    anthropic_api_key: str

    # models
    embedding_model: str = "text-embedding-3-small"
    synthesis_model: str = "claude-haiku-4-5"
    spacy_model: str = "en_core_web_sm"

    # synthesis
    synthesis_max_tokens: int = 1024

    # chunking
    chunk_size: int = 1500
    chunk_overlap: int = 150
    recursive_separators: list[str] = ["\n\n", "\n", ". "]
    reference_citation_threshold: float = 0.10

    # retrieval
    top_k: int = 5

    # storage
    chroma_path: str = "data/chroma"
    sqlite_db_path: str = "data/database.db"

    @property
    def sqlite_db_uri(self) -> str:
        return f"sqlite:///{self.sqlite_db_path}"


settings = Settings()
