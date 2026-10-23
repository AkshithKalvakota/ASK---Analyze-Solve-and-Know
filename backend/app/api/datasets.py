import os

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
from app.services.profiling import load_dataframe, profile_dataset, excel_sheet_names
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
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

UNREADABLE_FILE = "Couldn't read this file. Make sure it opens in Excel and is a .csv, .xlsx or .xls file."


def dataset_sheet(dataset: Dataset) -> str | None:
    """The Excel sheet the user picked (kept in the profile); None means the first sheet or a CSV."""
    return (dataset.profile_result or {}).get("sheet")


def read_dataset_file(dataset: Dataset):
    file_bytes = download_file(dataset.storage_key)
    try:
        return load_dataframe(file_bytes, dataset.content_type, dataset.filename, dataset_sheet(dataset))
    except Exception:
        raise HTTPException(status_code=400, detail=UNREADABLE_FILE)


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

    if not dataset.profile_result:
        raise HTTPException(status_code=400, detail="Run a data check before choosing a target.")

    if payload.target_column not in dataset.profile_result.get("dtypes", {}):
        raise HTTPException(status_code=400, detail="Column not found in dataset")

    if payload.problem_type_override:
        problem_type = payload.problem_type_override
    else:
        df = read_dataset_file(dataset)
        problem_type = detect_problem_type(df, payload.target_column)

    dataset.target_column = payload.target_column
    dataset.problem_type = problem_type
    db.commit()
    db.refresh(dataset)
    return dataset


# A plain def (not async): the B2 upload and database calls below are blocking,
# so FastAPI runs this in a worker thread instead of freezing every other request
@router.post("", response_model=DatasetOut)
def upload_dataset(
    project_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Use a .csv, .xlsx or .xls file.",
        )

    file_bytes = file.file.read()
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
    sheet: str | None = None,  # Excel only: switch to this sheet
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
    try:
        sheets = excel_sheet_names(file_bytes, dataset.content_type, dataset.filename)
    except Exception:
        raise HTTPException(status_code=400, detail=UNREADABLE_FILE)
    if sheet is not None and sheet not in sheets:
        raise HTTPException(status_code=400, detail="That sheet isn't in this file.")

    # Keep the user's earlier choice when re-checking; otherwise start with the first sheet
    previous_sheet = dataset_sheet(dataset)
    chosen = sheet or (previous_sheet if previous_sheet in sheets else (sheets[0] if sheets else None))
    try:
        df = load_dataframe(file_bytes, dataset.content_type, dataset.filename, chosen)
    except Exception:
        raise HTTPException(status_code=400, detail=UNREADABLE_FILE)

    result = profile_dataset(df)
    if sheets:
        result["sheets"] = sheets
        result["sheet"] = chosen

    # Another sheet means other data: the old target choice no longer applies
    # (no saved choice means the first sheet was being read)
    switched_sheet = sheet is not None and sheet != (previous_sheet or sheets[0])
    if switched_sheet or (dataset.target_column and dataset.target_column not in result["dtypes"]):
        dataset.target_column = None
        dataset.problem_type = None

    dataset.profile_result = result
    db.commit()
    db.refresh(dataset)
    return dataset