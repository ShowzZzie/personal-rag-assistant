from pydantic_settings import BaseSettings, SettingsConfigDict

MODEL_PRICING: dict[str, tuple[float, float]] = { # base rates, USD per MTok, as of 2026-08
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-1-20250805": (15.0, 75.0),
    "claude-opus-4-20250514": (15.0, 75.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
}


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
