import requests as rqst
from twilio.rest import Client
import json
import os

api_key = 'zzzzzzzzzz'
LAT = "z"
LONG = "z"
COUNT = 4
PRECIP_COMING = False

account_sid = "z"
auth_token = 'z'
client = Client(account_sid, auth_token)


parameters = {'lat': LAT, 'lon': LONG, 'cnt': COUNT, 'appid': api_key}

response = rqst.get("https://api.openweathermap.org/data/2.5/forecast", params=parameters)
response.raise_for_status()
weather_data = response.json()
for i in range(COUNT):
    num_weather_ids = len(weather_data['list'][i]['weather'])
    for x in range(num_weather_ids):
        weather_id = weather_data['list'][i]['weather'][x]['id']
        if int(weather_id) > 700:
            PRECIP_COMING = True


MESSAGE = (client.messages.create(
    body="You'll want an ☂️",
    from_="z",
    to="z",
))

if PRECIP_COMING:
    print("You'll want an ☂️")

    print(MESSAGE.status)
