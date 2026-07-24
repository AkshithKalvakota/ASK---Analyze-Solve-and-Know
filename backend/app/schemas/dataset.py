from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Optional, Any

class DatasetOut(BaseModel):
    id: UUID
    project_id: UUID
    filename: str
    content_type: str
    profile_result: Optional[dict[str, Any]] = None
    created_at: datetime

    class Config:
        from_attributes = True