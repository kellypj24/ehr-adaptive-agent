from pydantic import BaseSettings


class Settings(BaseSettings):
    FHIR_SERVER_URL: str = "https://hapi.fhir.org/baseR4"

    class Config:
        env_file = ".env"
