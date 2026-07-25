from sqlalchemy import Column, String, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
from app.db.base import Base

class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False, index=True)
    model_name = Column(String, nullable=False)
    storage_key = Column(String, nullable=False)
    preprocessing_log = Column(JSONB, nullable=False)
    metrics = Column(JSONB, nullable=False)
    all_results = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())