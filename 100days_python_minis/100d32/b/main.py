##################### Extra Hard Starting Project ######################

# 1. Update the birthdays.csv

# 2. Check if today matches a birthday in the birthdays.csv

# 3. If step 2 is true, pick a random letter from letter templates and replace the [NAME] with the person's actual name from birthdays.csv

# 4. Send the letter generated in step 3 to that person's email address.

import pandas as pd
import datetime as dt
import random
import pathlib
import smtplib

letter = random.choice(list(pathlib.Path
               ("C:/Education/Git_workspace/100days_python_minis/100d32/b/letter_templates")
               .iterdir()))



birthdays = pd.read_csv("birthdays.csv")
now = dt.datetime.now()
print(birthdays)
df = pd.DataFrame()
df[['bday_name','email']] = (birthdays[(birthdays['year']== now.year)
                                       & (birthdays['day'] == now.day)
                                       & (birthdays['month'] == now.month)][['name','email']])

for index, row in df.iterrows():
    with open(letter) as file:
        message = file.read()
        message = message.replace('[Name]',row['bday_name'])
        with smtplib.SMTP("smtp.gmail.com") as connection:
            connection.starttls()
            connection.login(user=my_email, password=password)
            connection.sendmail(from_addr=my_email,
                                to_addrs=row['email'],
                                msg=f"Subject:Happy Birthday!!!\n\n{message}")

print(message)
