from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    clerk_secret_key: str
    b2_access_key_id: str
    b2_secret_access_key: str
    b2_endpoint_url: str
    b2_bucket_name: str

    class Config:
        env_file = ".env"

settings = Settings()