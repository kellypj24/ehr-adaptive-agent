from pydantic_settings import BaseSettings
from pydantic import ConfigDict


class Settings(BaseSettings):
    FHIR_SERVER_URL: str = "https://hapi.fhir.org/baseR4"

    model_config = ConfigDict(env_file=".env")
