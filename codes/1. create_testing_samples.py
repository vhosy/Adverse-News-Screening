import pandas as pd
import requests
from news_extraction_helper import create_news_df

api_key = ''
#get all articles about fraud
url = f'https://newsapi.org/v2/everything?q="fraud"&excludeDomains=yahoo.com&apiKey={api_key}'
response = requests.get(url)
data = response.json()
fraud_df= create_news_df (data)

#get all articles about tax evasion
url = f'https://newsapi.org/v2/everything?q="tax evasion"&excludeDomains=yahoo.com&apiKey={api_key}'
response = requests.get(url)
data = response.json()
tax_evade_df= create_news_df (data)

#get all articles about financial growth (positive news)
#there were several articles about interest rates that had more neutral tone so they
#were manually filtered to a neutral_df
url = f'https://newsapi.org/v2/everything?q="financial growth"&excludeDomains=yahoo.com&apiKey={api_key}'
response = requests.get(url)
data = response.json()
positive_df = create_news_df (data)

fraud_df.to_csv('data/fraud_df.csv')
tax_evade_df.to_csv('data/tax_evade_df.csv')
positive_df.to_csv('data/positive_df.csv')
