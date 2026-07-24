from pydantic import BaseModel
from datetime import datetime
from uuid import UUID

class ProjectCreate(BaseModel):
    name: str

class ProjectOut(BaseModel):
    id: UUID
    name: str
    created_at: datetime

    class Config:
        from_attributes = True