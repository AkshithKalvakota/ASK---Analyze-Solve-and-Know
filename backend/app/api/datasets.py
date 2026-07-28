from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID
from pydantic import BaseModel

from app.models.trained_model import TrainedModel
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.models.project import Project
from app.schemas.dataset import DatasetOut
from app.api.deps import get_current_user_id
from app.services.storage import upload_file, download_file
from app.services.profiling import load_dataframe, profile_dataset
from app.services.target_detection import detect_problem_type

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


class TargetSelection(BaseModel):
    target_column: str
    problem_type_override: str | None = None


@router.post("/{dataset_id}/target", response_model=DatasetOut)
def set_target_column(
    project_id: UUID,
    dataset_id: UUID,
    payload: TargetSelection,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    dataset = db.execute(stmt).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    if payload.target_column not in dataset.profile_result.get("dtypes", {}):
        raise HTTPException(status_code=400, detail="Column not found in dataset")

    if payload.problem_type_override:
        problem_type = payload.problem_type_override
    else:
        file_bytes = download_file(dataset.storage_key)
        df = load_dataframe(file_bytes, dataset.content_type)
        problem_type = detect_problem_type(df, payload.target_column)

    dataset.target_column = payload.target_column
    dataset.problem_type = problem_type
    db.commit()
    db.refresh(dataset)
    return dataset


@router.post("", response_model=DatasetOut)
async def upload_dataset(
    project_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
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


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(
    project_id: UUID,
    dataset_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    dataset = db.execute(stmt).scalar_one_or_none()
    if dataset:
        db.query(TrainedModel).filter(TrainedModel.dataset_id == dataset_id).delete(synchronize_session=False)
        db.delete(dataset)
        db.commit()


@router.post("/{dataset_id}/profile", response_model=DatasetOut)
def profile_dataset_endpoint(
    project_id: UUID,
    dataset_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(Dataset).where(Dataset.id == dataset_id, Dataset.project_id == project_id)
    dataset = db.execute(stmt).scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_bytes = download_file(dataset.storage_key)
    df = load_dataframe(file_bytes, dataset.content_type)
    result = profile_dataset(df)

    dataset.profile_result = result
    db.commit()
    db.refresh(dataset)
    return dataset