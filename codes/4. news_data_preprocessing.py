from news_extraction_helper import get_fr_gnews, get_redirected_urls, get_article_text

#extract all cnbc news from google rss feed between oct 2024 to jan 2025
df = get_fr_gnews('cnbc.com', '2024-10-01', '2025-01-31')

# get the redirected url from google news
df['redirected_urls'] =  get_redirected_urls(df['urls'])

#extract the news article from the url 
df['text'] = get_article_text(df['redirected_urls'])

#remove all articles where links are for videos, links outdated
df = df[df['text'].notna()]

#check and drop if there are duplicates
df = df.drop_duplicates()

#save
df.to_csv('data/cnbc_news_df.csv', index = False)
