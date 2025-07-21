from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from models import Todos, Users
from database import SessionLocal
from typing import Annotated
from sqlalchemy.orm import Session
from starlette import status
from .auth import get_current_user, auth_401_error
from passlib.context import CryptContext


router = APIRouter(
    prefix='/user',
    tags=['user']
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]
bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')


class NewPass(BaseModel):
    new_pass: str = Field(min_length=3, max_length=100)

@router.get('/info', status_code=status.HTTP_200_OK)
async def get_user(
    user: user_dependency,
    db: db_dependency
    ):

    if not user:
        auth_401_error()
    
    user_model = db.query(Users).filter(Users.id == user.get('id')).first()
    return user_model


@router.put('/password', status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    user: user_dependency,
    db: db_dependency,
    new_pass: NewPass
):
    if not user:
        auth_401_error()

    user_model = db.query(Users).filter(Users.id == user.get('id')).first()

    user_model.hash_pass = bcrypt_context.hash(new_pass.new_pass)
    db.add(user_model)
    db.commit()


@router.put('/update_phone', status_code=status.HTTP_204_NO_CONTENT)
async def update_user(
    user: user_dependency,
    db: db_dependency,
    phone_entry: dict = {'New Phone #':''}
    ):
    
    if not user:
        auth_401_error()

    user_model = db.query(Users).filter(Users.id == user.get('id')).first()
    user_model.phone_number = phone_entry['New Phone #']
    db.add(user_model)
    db.commit()

