The objective of this project was to develop a system to automatically screen publicly available news articles and identify entities (individuals or corporations) potentially involved in financial crime, scandals, or sanctions.

Below are the steps taken to achieve this:
1.  Extracted news articles from CNBC website
2.  Identified individuals and corporations mentioned in the news article, tagged them and assigned Wikidata link to the individual/corporation
3.  Used sentiment classification model to classify if news sentiment is positive, neutral or negative
4.  Used zero shot classification to label the financial crime topic of the news article
5.  Load all output to Tableau dashboard: 

Ideally, with only adverse financial news, all tagged with their topic, relevance score and a short summary, all stored in a single dashboard for users’ quick and easy perusal:

-  No need to manually scan news website for the relevant articles
-  Short summary of article allows for faster reading, URL links still available to read full article
-  Relevance score helps users quickly identify which are the more important articles to focus on
-  Update watchlist of people or organizations mentioned
-  Learn from others’ mistakes and review policies and workflows to reduce risks of similar situation occurring

Please refer to the 'Adverse News Screening for Financial Crime Surveillance.docx' for more details.
