from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class DatasetOut(BaseModel):
    id: UUID
    project_id: UUID
    filename: str
    content_type: str
    created_at: datetime

    class Config:
        from_attributes = True