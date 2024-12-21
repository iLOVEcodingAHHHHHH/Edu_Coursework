import requests

pixela_endpoint = "https://pixe.la/v1/users"

USERNAME = "moogoat"
TOKEN = "imatoken111111111"

user_params = {
    "token": "imatoken111111111",
    "username": "moogoat",
    "agreeTermsOfService": "yes",
    "notMinor": "yes",
}

#response = requests.post(url=pixela_endpoint, json=user_params)
#print(response.text)

graph_endpoint = f"{pixela_endpoint}/{USERNAME}/graphs"

graph_config = {
    "id": "assignment",
    "name": "test",
    "unit": "units",
    "type": "float",
    "color": "green"
}

headers = {
    "X-USER-TOKEN": TOKEN
}

#requests.get(url=graph_endpoint, json=graph_config, headers=headers)

pixel_creation_endpoint = f"{pixela_endpoint}/{USERNAME}/graphs/graph1"
pixel_data = {
    "date": "20241111",
    "quantity": "10",
}

response = requests.post(url=pixel_creation_endpoint, json=pixel_data, headers=headers)
print(response.text)