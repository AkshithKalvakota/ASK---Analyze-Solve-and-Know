from pydantic import BaseModel
from datetime import datetime
from uuid import UUID
from typing import Any

class TrainedModelOut(BaseModel):
    id: UUID
    dataset_id: UUID
    model_name: str
    metrics: dict[str, Any]
    all_results: dict[str, Any]
    created_at: datetime

    class Config:
        from_attributes = True