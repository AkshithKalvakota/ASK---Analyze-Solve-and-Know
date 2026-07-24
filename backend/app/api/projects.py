from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import select
from uuid import UUID

from app.db.session import SessionLocal
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectOut
from app.api.deps import get_current_user_id

router = APIRouter(prefix="/projects", tags=["projects"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("", response_model=ProjectOut)
def create_project(
    payload: ProjectCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    project = Project(name=payload.name, user_id=user_id)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project

@router.get("", response_model=list[ProjectOut])
def list_projects(
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.user_id == user_id)
    return db.execute(stmt).scalars().all()

@router.delete("/{project_id}", status_code=204)
def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(Project).where(Project.id == project_id, Project.user_id == user_id)
    project = db.execute(stmt).scalar_one_or_none()
    if project:
        db.delete(project)
        db.commit()