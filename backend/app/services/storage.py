import boto3
from app.core.config import settings

s3_client = boto3.client(
    "s3",
    endpoint_url=settings.b2_endpoint_url,
    aws_access_key_id=settings.b2_access_key_id,
    aws_secret_access_key=settings.b2_secret_access_key,
)

def upload_file(file_bytes: bytes, key: str, content_type: str) -> str:
    """
    Uploads a file to B2 and returns the storage key (path) used.
    'key' should be a unique path like f"{project_id}/{filename}"
    """
    s3_client.put_object(
        Bucket=settings.b2_bucket_name,
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return key

def get_download_url(key: str, expires_in: int = 3600) -> str:
    """
    Generates a temporary signed URL to download a private file.
    """
    return s3_client.generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.b2_bucket_name, "Key": key},
        ExpiresIn=expires_in,
    )