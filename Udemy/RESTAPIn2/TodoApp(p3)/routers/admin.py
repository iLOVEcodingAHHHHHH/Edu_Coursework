from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from models import Todos
from database import SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
from starlette import status
from .auth import get_current_user, auth_401_error
from.todos import todo_404_error


router = APIRouter(
    prefix='/admin',
    tags=['admin']
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]


@router.get("/todo", status_code=status.HTTP_200_OK)
async def read_all(user: user_dependency, db: db_dependency):
    if not user or user.get('user_role') != 'admin':
        auth_401_error()
    return db.query(Todos).all()


@router.delete("/todo/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_todo(
    user: user_dependency,
    db: db_dependency,
    todo_id: int = Path(gt=0)
    ) -> None:
    
    if not user or user.get('user_role') != 'admin':
        auth_401_error()
    
    todo_model = db.query(Todos).filter(Todos.id == todo_id).first()
    
    if not todo_model: 
        todo_404_error()
    db.delete(todo_model)
    db.commit()