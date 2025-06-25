from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from models import Todos
from typing import Annotated
from sqlalchemy.orm import Session
from database import SessionLocal
from starlette import status
from .auth import auth_401_error, get_current_user


router = APIRouter(
    prefix = '/tasks',
    tags = ['todos']
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

class TodoRequest(BaseModel):
    title: str = Field(min_length=3)
    description: str = Field(min_length=3, max_length=100)
    priority: int = Field(gt=0, lt=6)
    complete: bool

def todo_404_error():
    raise HTTPException(status_code=404, detail='Todo not found.')


@router.get("/", status_code=status.HTTP_200_OK)
async def read_all(user: user_dependency, db: db_dependency):
    if not user: auth_401_error()
    return db.query(Todos).filter(Todos.owner_id == user.get('id')).all()

@router.get("/todo/{todo_id}", status_code=status.HTTP_200_OK)
async def read_todo(user: user_dependency,
                    db: db_dependency,
                    todo_id: int = Path(gt=0)):
    
    if not user: auth_401_error()
    todo_model = db.query(Todos).filter(Todos.id == todo_id,
                                        Todos.owner_id == user.get('id')).first()
    if todo_model:
        return todo_model
    todo_404_error()

@router.post("/todo", status_code=status.HTTP_201_CREATED)
async def create_todo(
    user: user_dependency,
    db: db_dependency,
    todo_request: TodoRequest
    ):
    if not user:
        auth_401_error()
    todo_model = Todos(**todo_request.model_dump(), owner_id=user.get('id'))
    
    db.add(todo_model)
    db.commit()

@router.put("/todo/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def update_todo(
    user: user_dependency,
    db: db_dependency,
    todo_request: TodoRequest,
    todo_id: int = Path(gt=0)
    ):
    if not user: auth_401_error()
    
    todo_model = db.query(Todos).filter(Todos.id == todo_id,
                                        Todos.owner_id == user.get('id')).first()
    if not todo_model: todo_404_error()
    
    todo_model.title = todo_request.title
    todo_model.description = todo_request.description
    todo_model.priority = todo_request.priority
    todo_model.complete = todo_request.complete

    db.add(todo_model)
    db.commit()

@router.delete("/todo/{todo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_todo(user: user_dependency,
                      db: db_dependency,
                      todo_id: int = Path(gt=0)):
    if not user: auth_401_error()

    todo_model = db.query(Todos).filter(Todos.id == todo_id,
                                        Todos.owner_id == user.get('id')).first()
    if not todo_model: todo_404_error()

    db.delete(todo_model)
    db.commit()
    