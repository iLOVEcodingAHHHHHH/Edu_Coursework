import requests
import json
import os
import pandas as pd
from datetime import date as dt

YESTERDAY = (dt.today() - pd.Timedelta('1d'))
DAY_PRIOR = (YESTERDAY - pd.Timedelta('1d'))


STOCK_NAME = "TSLA"
COMPANY_NAME = "Tesla Inc"

STOCK_ENDPOINT = "https://www.alphavantage.co/query"
NEWS_ENDPOINT = "https://newsapi.org/v2/everything"

alpha_params={
    'function': 'TIME_SERIES_DAILY',
    'symbol': STOCK_NAME,
    'apikey': os.environ.get('alpha_api_key'),
}

news_params={
    'qInTitle': COMPANY_NAME,
    'from_param': DAY_PRIOR,
    'to': YESTERDAY,
    'language': 'en',
    'sort_by': 'relevancy',
    'page':1,
    'apikey': os.environ.get('news_api_key')
}

    ## STEP 1: Use https://www.alphavantage.co/documentation/#daily
# When stock price increase/decreases by 5% between yesterday and the day before yesterday then print("Get News").

#TODO 1. - Get yesterday's closing stock price. Hint: You can perform list comprehensions on Python dictionaries. e.g. [new_value for (key, value) in dictionary.items()]

av_response = requests.get(STOCK_ENDPOINT,params=alpha_params)
av_response.raise_for_status()
print(av_response.json())

tsla_yesterday = float(av_response.json()['Time Series (Daily)'][str(YESTERDAY)]['4. close'])
tsla_dayprior = float(av_response.json()['Time Series (Daily)'][str(DAY_PRIOR)]['4. close'])
tsla_movement = (tsla_yesterday-tsla_dayprior)/tsla_dayprior
print(tsla_movement)

if abs(tsla_movement) > 0.001:

    news_response = requests.get(NEWS_ENDPOINT,params=news_params)
    news_response.raise_for_status()

    for article in news_response.json()['articles']:
        print(article['title'])

    three_articles = news_response.json()['articles'][:3]

print(three_articles)
#TODO 2. - Get the day before yesterday's closing stock price

#TODO 3. - Find the positive difference between 1 and 2. e.g. 40 - 20 = -20, but the positive difference is 20. Hint: https://www.w3schools.com/python/ref_func_abs.asp

#TODO 4. - Work out the percentage difference in price between closing price yesterday and closing price the day before yesterday.

#TODO 5. - If TODO4 percentage is greater than 5 then print("Get News").

    ## STEP 2: https://newsapi.org/
    # Instead of printing ("Get News"), actually get the first 3 news pieces for the COMPANY_NAME.

#TODO 6. - Instead of printing ("Get News"), use the News API to get articles related to the COMPANY_NAME.

#TODO 7. - Use Python slice operator to create a list that contains the first 3 articles. Hint: https://stackoverflow.com/questions/509211/understanding-slice-notation


    ## STEP 3: Use twilio.com/docs/sms/quickstart/python
    #to send a separate message with each article's title and description to your phone number.

#TODO 8. - Create a new list of the first 3 article's headline and description using list comprehension.

#TODO 9. - Send each article as a separate message via Twilio.



#Optional TODO: Format the message like this:
"""
TSLA: 🔺2%
Headline: Were Hedge Funds Right About Piling Into Tesla Inc. (TSLA)?.
Brief: We at Insider Monkey have gone over 821 13F filings that hedge funds and prominent investors are required to file by the SEC The 13F filings show the funds' and investors' portfolio positions as of March 31st, near the height of the coronavirus market crash.
or
"TSLA: 🔻5%
Headline: Were Hedge Funds Right About Piling Into Tesla Inc. (TSLA)?.
Brief: We at Insider Monkey have gone over 821 13F filings that hedge funds and prominent investors are required to file by the SEC The 13F filings show the funds' and investors' portfolio positions as of March 31st, near the height of the coronavirus market crash.
"""

