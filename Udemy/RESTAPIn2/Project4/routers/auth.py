from datetime import datetime, timedelta, timezone
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Users
from passlib.context import CryptContext
from starlette import status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from jose import jwt, JWTError


router = APIRouter(
    prefix='/auth',
    tags=['auth']
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

SECRET_KEY = "asdf123"
ALGORITHM = "HS256"

bcrypt_context = CryptContext(schemes=['bcrypt'], deprecated='auto')
oauth2_bearer = OAuth2PasswordBearer(tokenUrl='auth/token')

db_dependency = Annotated[Session, Depends(get_db)]

def auth_401_error():
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                        detail='Could not validate user.')


class CreateUserRequest(BaseModel):
    username: str
    email: str
    first_name: str
    last_name: str
    phone_number: str
    password: str
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str



def authenticate_user(username: str, password: str, db):
    user = db.query(Users).filter(Users.username == username).first()

    if not user:
        return False
    if not bcrypt_context.verify(password, user.hash_pass):
        return False
    return user


def create_access_token(
    username: str,
    user_id: int,
    role: str,
    expires_delta: timedelta
    ) -> str:

    encode = {'sub': username, 'id': user_id, 'role': role}
    expires = datetime.now(timezone.utc) + expires_delta
    encode.update({'exp': expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: Annotated[str, Depends(oauth2_bearer)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        user_role: str = payload.get('role')

        if not (username and user_id):
            auth_401_error()

        return {'username': username, 'id': user_id, 'user_role': user_role}
    
    except JWTError:
        auth_401_error()


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(
    db: db_dependency,
    create_user_request: CreateUserRequest
    ):
    
    create_user_model = Users(
        email=create_user_request.email,
        username=create_user_request.username,
        first_name=create_user_request.first_name,
        last_name=create_user_request.last_name,
        role=create_user_request.role,
        phone_number=create_user_request.phone_number,
        hash_pass=bcrypt_context.hash(create_user_request.password),
        is_active=True
    )

    db.add(create_user_model)
    db.commit()

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: db_dependency
    ) -> dict[str, str]:

    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        auth_401_error()

    token = create_access_token(user.username, user.id,
                                user.role, timedelta(minutes=20))
    return {'access_token': token, 'token_type': 'bearer'}