import requests
import pandas as pd
import newspaper
from gnews import GNews
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
from datetime import datetime, timedelta

options = Options()
options.add_argument('--headless')
driver = webdriver.Firefox(options=options)

API_key = ''

def generate_date_range(start_date, end_date):
    """
    Convert string of start_date and end_date to datetime objects and generate 
    a list of all dates in between
    
    Args:
         start_date (str): the start_date in YYYY-MM-DD format
         end_date (str): the end_date in YYYY-MM-DD format
         
     Returns:
         list: the list of dates in date format between the start_date and end_date
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Generate a list of dates between the two dates
    date_list = [(start + timedelta(days=i)).strftime("%Y-%m-%d") 
                 for i in range((end - start).days + 1)]
    date_list = [datetime.strptime(date, "%Y-%m-%d") for date in date_list]
      
    return date_list

# =============================================================================
# #News API doesn't get from all website, is only a curated list and can only search 1 month back
# def get_fr_newsapi(domain, fr_date, to_date):
#     
#     dates = generate_date_range(fr_date, to_date)
#     news_df = []
#     
#     for date in dates:
#         date1 = (date + timedelta(days=1)).strftime('%Y-%m-%d')
#         date = date.strftime('%Y-%m-%d')
#         
#         url = f'https://newsapi.org/v2/everything?domains={domain}&from={date}&to={date1}&apiKey={API_key}'
# 
#         response = requests.get(url)
#         data = response.json()
# 
#         df = pd.DataFrame()
#         df['url'] =  [x['url'] for x in data['articles'] ]
#         df['pub_date'] =  [datetime.strptime(x['publishedAt'],"%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d") 
#                            for x in data['articles'] ]
#         
#         df = df[df['pub_date']==date]
#         news_df.append(df)
#     
#     return news_df
# 
# def get_fr_newspaper4k (domain):
#     news = newspaper.build(domain, number_threads=3)
#     article_urls = [article.url for article in news.articles]
# 
#     article_urls = pd.DataFrame(article_urls)
#     article_urls = article_urls.rename(columns = {0:'url'})
#     
#     return article_urls
# =============================================================================



def get_fr_gnews(domain, fr_date, to_date):
    """
    get urls of news article fr particular domain using GNews library
    
    Args:
         fr_date (str): start date of search in YYYY-MM-DD format
         to_date (str): end_date of search in YYYY-MM-DD format
         domain (str): domain of news website
        
     Returns:
         pandas dataframe: dataframe of published date and url. 
         url is the google news url, i.e. https://news.google.com/rss/articles/...

    """
    dates = generate_date_range(fr_date, to_date)
    
    news_df = []
    for date in dates:
        # GNews function does not have a single date option, so we will extract 
        #between date and date + 1 day
        date1 = date + timedelta(days=1)
        google_news = GNews(start_date= (date.year, date.month, date.day), 
                            end_date= (date1.year, date1.month, date1.day))
        
        news = google_news.get_news_by_site(domain)
        
        #extract url and pub_date from news dict and store as df
        temp = pd.DataFrame()
        temp['urls']= [x['url'] for x in news]
        temp['pub_date']=  [datetime.strptime(x['published date'] , 
                                       "%a, %d %b %Y %H:%M:%S %Z").date() for x in news]
        
        #filter back to current date
        temp = temp[temp['pub_date']==date.date()]
        news_df.append(temp)
        
    news_df = pd.concat(news_df)
    
    return news_df


def get_redirected_urls(urls):
    """
    get redirected urls from google news feed url
    
    Args:
         urls (list): the google news feed url
        
     Returns:
         list: the list of urls of the news website after redirected
    """
    redirected_urls = []
    for i in urls:
        driver.get(i)
        #pause for 3 sec to ensure fully redirected
        time.sleep(3)  
        url=driver.current_url
        redirected_urls.append(url)
    return redirected_urls
        
def get_article_text(urls):
    """
    # get the article from the news website url, returns empty if cannot scrape
    
    Args:
         urls (list): the news website url
        
     Returns:
         list: the list of article text
    """
    articles = []
    for url in urls:
        #try extracting text, sometime content on the url could have been remove 
        #or url is video only, so no text, append as blank 
        try:
           article = newspaper.article(url)
           article_text = article.text
           articles.append(article_text)
        except Exception as e:
           articles.append('') 
    
    return articles

def create_news_df (data):
    """
    create dataset with url and published date from dictionary created from News API
    
    Args:
         data (dict): dictionary created from News API
        
     Returns:
         df: pandas dataframe with url and published date
    """
    df = pd.DataFrame()
    df['url'] = [x['url'] for x in data['articles']]
    df['pub_date'] = [x['publishedAt'] for x in data['articles']]

    df['text'] = get_article_text(df['url'])
    
    #keep only rows that have text
    df = df[df['text']!='']
    
    return df
    