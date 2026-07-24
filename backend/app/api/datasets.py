from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.models.project import Project
from app.schemas.dataset import DatasetOut
from app.api.deps import get_current_user_id
from app.services.storage import upload_file

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["datasets"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

ALLOWED_TYPES = {
    "text/csv",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}

@router.post("", response_model=DatasetOut)
async def upload_dataset(
    project_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    # Confirm the project exists and belongs to this user
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="File must be CSV or Excel")

    file_bytes = await file.read()
    storage_key = f"{project_id}/{file.filename}"
    upload_file(file_bytes, storage_key, file.content_type)

    dataset = Dataset(
        project_id=project_id,
        filename=file.filename,
        storage_key=storage_key,
        content_type=file.content_type,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset

@router.get("", response_model=list[DatasetOut])
def list_datasets(
    project_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(Dataset).where(Dataset.project_id == project_id)
    return db.execute(stmt).scalars().all()