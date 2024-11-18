import json
import requests
import datetime as dt
import smtplib
import time



MY_LAT = 30.151573
MY_LNG = -81.617610
reset = 0

parameters = {
    "lat": MY_LAT,
    "lng": MY_LNG,
    "formatted": 0,
}

iss_response = requests.get("http://api.open-notify.org/iss-now.json")
iss_response.raise_for_status()

data = iss_response.json()
iss_longlat_tuple = (data['iss_position']['longitude'],data['iss_position']['latitude'])

now = str(dt.datetime.now()).split(" ")[1].split(":")[0]


sun_response = requests.get("https://api.sunrise-sunset.org/json", params=parameters)
sun_response.raise_for_status()
sun_data = sun_response.json()
with open('txt.txt', 'w') as file:
    json.dump(sun_data, file)
sunrise_hour = int(sun_data['results']['sunrise'].split("T")[1].split(":")[0])
sunset_hour = int(sun_data['results']['sunset'].split("T")[1].split(":")[0])

def do_the_thing():
    if now < sunrise_hour or now > sunset_hour:
        global x
        x = 1
        if MY_LAT-5 <= iss_longlat_tuple[0] <= MY_LAT+5 and MY_LNG-5 <= iss_longlat_tuple[1] <= MY_LNG+5:
            with smtplib.SMTP("smtp.gmail.com") as connection:
                connection.starttls()
                connection.login(user="XXXXX", password="XXXX")
                connection.sendmail(from_addr='XXXXXX@gmail.com',
                                    to_addrs='XXXXX@gmail.com',
                                    msg="ISS_ALERT!\n\nIt's that time!")

while True:
    time.sleep(60)
    print('pretend_to_do_the_thing()')
    if dt.datetime.now().hour == 00:
        global reset
        reset = 0