import pandas as pd
import numpy as np
from classification_helper import predict_sentiment, predict_topic

#load data
df = pd.read_csv('data/cnbc_news_df.csv')

#the categories to classify data into
candidate_labels = ['fraud', 'tax evasion', 'scam', 'other financial crime', 'non financial crime']

#predict the sentiment and financial crime topic with best model selected from testing
df['sentiment_label'], df['sentiment_score'], df['news_summary'] = predict_sentiment ("ProsusAI/finbert", df['text'])
df['topic_label'], df['topic_score'], df['top_topic'], df['top_score'], summary =\
    predict_topic ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", df['text'], candidate_labels)

#extract the news domain from the url
df['domain'] = df['redirected_urls'].str.extract(r'www\.(.*?)\.com')
#calculate the relevance score
df['relevance_score'] = np.where(df['sentiment_label'] == 'positive',  -df['sentiment_score'] * df['top_score'],
                                 np.where(df['sentiment_label'] == 'neutral',  0.5 * df['sentiment_score'] * df['top_score'],
                                           df['sentiment_score'] * df['top_score']))
df['relevance_score'] = np.where(df['top_topic'] == 'non financial crime', np.nan, df['relevance_score'])

df.to_csv('data/cnbc_news_results_df.csv', index = False)
