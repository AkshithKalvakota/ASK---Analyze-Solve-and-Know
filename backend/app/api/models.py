from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID
from pydantic import BaseModel
from typing import Any
import joblib
from io import BytesIO
from app.db.session import SessionLocal
from app.models.dataset import Dataset
from app.models.project import Project
from app.models.trained_model import TrainedModel
from app.schemas.trained_model import TrainedModelOut
from app.api.deps import get_current_user_id
from app.services.storage import download_file, upload_file
from app.services.profiling import load_dataframe
from app.services.preprocessing import preprocess
from app.services.automl import train_and_select_best

router = APIRouter(prefix="/projects/{project_id}/datasets/{dataset_id}/models", tags=["models"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=TrainedModelOut)
def train_model(
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

    if not dataset.target_column or not dataset.problem_type:
        raise HTTPException(status_code=400, detail="Target column not set for this dataset")

    file_bytes = download_file(dataset.storage_key)
    df = load_dataframe(file_bytes, dataset.content_type)

    X, y, preprocessing_log = preprocess(df, dataset.target_column)
    try:
        result = train_and_select_best(X, y, dataset.problem_type)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    model_storage_key = f"{project_id}/{dataset_id}/model_{result['best_model_name']}.joblib"
    upload_file(result["model_bytes"], model_storage_key, "application/octet-stream")

    trained_model = TrainedModel(
        dataset_id=dataset_id,
        model_name=result["best_model_name"],
        storage_key=model_storage_key,
        preprocessing_log=preprocessing_log,
        metrics=result["best_model_metrics"],
        all_results=result["all_results"],
    )
    db.add(trained_model)
    db.commit()
    db.refresh(trained_model)
    return trained_model

@router.get("", response_model=list[TrainedModelOut])
def list_models(
    project_id: UUID,
    dataset_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(TrainedModel).where(TrainedModel.dataset_id == dataset_id)
    return db.execute(stmt).scalars().all()

class PredictionInput(BaseModel):
    values: dict[str, Any]

@router.post("/{model_id}/predict")
def predict(
    project_id: UUID,
    dataset_id: UUID,
    model_id: UUID,
    payload: PredictionInput,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(TrainedModel).where(
        TrainedModel.id == model_id, TrainedModel.dataset_id == dataset_id
    )
    trained_model = db.execute(stmt).scalar_one_or_none()
    if not trained_model:
        raise HTTPException(status_code=404, detail="Model not found")

    model_bytes = download_file(trained_model.storage_key)
    model = joblib.load(BytesIO(model_bytes))

    from app.services.preprocessing import apply_preprocessing_to_row
    X_row = apply_preprocessing_to_row(payload.values, trained_model.preprocessing_log)

    prediction = model.predict(X_row)[0]

    result: dict[str, Any] = {}
    target_categories = trained_model.preprocessing_log.get("target_categories")
    if target_categories:
        result["prediction"] = target_categories[int(prediction)]
    else:
        result["prediction"] = float(prediction)

    return result

@router.get("/{model_id}/input-schema")
def get_input_schema(
    project_id: UUID,
    dataset_id: UUID,
    model_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    stmt = select(TrainedModel).where(
        TrainedModel.id == model_id, TrainedModel.dataset_id == dataset_id
    )
    trained_model = db.execute(stmt).scalar_one_or_none()
    if not trained_model:
        raise HTTPException(status_code=404, detail="Model not found")

    stmt = select(Dataset).where(Dataset.id == dataset_id)
    dataset = db.execute(stmt).scalar_one_or_none()

    dtypes = dataset.profile_result.get("dtypes", {})
    original_columns = [
        col for col in dtypes.keys() if col != dataset.target_column
    ]

    return {"fields": [{"name": col, "type": dtypes[col]} for col in original_columns]}