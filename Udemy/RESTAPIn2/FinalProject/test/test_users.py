from .utils import *
from routers.users import get_db, get_current_user
from fastapi import status
from models import Users

app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[get_current_user] = override_get_current_user

def test_return_user(test_user):
    response = client.get('/user/info')
    assert response.status_code == 200
    assert response.json()['id'] == 1
    # and json()['blah'] for each k,v


def test_change_password_success(test_user):
    response = client.put(
        '/user/password',
        json={
            "new_pass": "newpassword"
        }
    )

    assert response.status_code == 204

    
def test_change_phone(test_user):
    response = client.put(
        '/user/update_phone',
        json={"New Phone #": '(222)-222-2222'}
        )
    
    assert response.status_code == 204