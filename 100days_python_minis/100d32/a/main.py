import smtplib
import datetime as dt
import random

day_dict = {1:"",2:"",3:"",4:"Friday",5:"",6:"",7:""}
day_of_week = dt.datetime.now().weekday()

lines=open('quotes.txt').read().splitlines()
friday_quote = random.choice(lines)

my_email = "askewprogramming@gmail.com"
password = "cnhx hjpj egpy plat"
if day_dict[day_of_week] == "Friday":

    with smtplib.SMTP("smtp.gmail.com") as connection:
        connection.starttls()
        connection.login(user=my_email, password=password)
        connection.sendmail(from_addr=my_email,
                           to_addrs="bad.moogoat@gmail.com",
                           msg=f"Subject:hello\n\n{friday_quote}")

