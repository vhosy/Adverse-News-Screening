import pandas as pd
from transformers import pipeline, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
import statistics
import nltk
from collections import defaultdict

def chunk_string(text, max_words=4000):
    """
     cut up the text into chunks of specified number of words

     Args:
         text (str): the text to be chunked
         max_words (int): the maximum number of words in each chunk

     Raises:
         TypeError: If 'text' is not a string
         
     Returns:
         list: the list of chunked strings

     """
    if not isinstance(text, str):
        raise TypeError ('text needs to be a string')

    words =  nltk.word_tokenize(text)
    chunks = []
    current_chunk = []
    
    #add word to current_chunk if the length is shorter than max word specified
    #else add current_chunk list to chunk list
    for word in words:
        if len(current_chunk + [word]) <= max_words:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]

    chunks.append(' '.join(current_chunk))
    return chunks

def predict_sentiment (model, text_list):  
    """
     predicts sentiment of text using text classification model from hugging face

     Args:
         model (str): hugging face model to be used, e.g. "ProsusAI/finbert"
         text_list (list): the list of text to perform sentiment classification
        
     Returns:
         sentiment_label (list): the sentiment (negative, neutral or positive) of the text
         sentiment_score (list): the confidence score of the sentiment

     """
     #initiate the sentiment classification model
    if model == "yiyanghkust/finbert-tone":
        finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
        tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

        pipe = pipeline("sentiment-analysis", model=finbert, tokenizer=tokenizer)
    
    else:
        pipe = pipeline("text-classification", model= model)
    
    #initiate summariser model
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    word_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    
    summary_list = []
    sentiment_label = []
    sentiment_score = []
    #for text in text_list, if the text has length > 1024, chunk up the text, 
    #summarise and finally predict with sentiment classification model
    #get final label from majority vote and average the score amongst the majority vote
    for text in text_list:
        if len(word_tokenizer.tokenize(text)) > 1024:
            chunks = chunk_string(text, max_words = 800)
            summary = [summarizer(chunk, max_length=514, min_length=min(100, len(word_tokenizer.tokenize(text))), do_sample=False)[0]["summary_text"] for chunk in chunks]
            predict = [pipe(s) for s in summary]
            sentiment = [x[0]['label'] for x in predict]
            score =  [x[0]['score'] for x in predict]
        
            frequency = pd.Series(sentiment).value_counts()
            sentiment_index = [i for i, j in enumerate(sentiment) if j == frequency.idxmax()]
            score = statistics.mean([score[x] for x in sentiment_index])
        
            sentiment_label.append(frequency.idxmax())
            sentiment_score.append(score)
            summary_list.append(summary)
        
        else:
            text = " ".join(text.split())
            summary = summarizer(text, max_length=514, min_length=min(100,len(word_tokenizer.tokenize(text))), do_sample=False)[0]["summary_text"]
            predict = pipe(summary)
            sentiment_label.append(predict[0]['label'])
            sentiment_score.append(predict[0]['score'])
            summary_list.append(summary)
    
    return sentiment_label, sentiment_score, summary

def predict_topic (model, text_list, candidate_labels): 
    """
     predicts topic of text using zero shot classification from hugging face, 
     based on specified topics input by user

     Args:
         model (str): hugging face model to be used, e.g. "facebook/bart-large-mnli"
         text_list (list): the list of text to perform topic modelling
         candidate_labels (list): the list of topics given by user
        
     Returns:
         topic_label (list): the list of topic_labels for each text
         topic_score (list): the list of confidence score for each text
         top_topic (list): the top topic for each text
         top_score (list): the top score of the top topic for each text
         
     """ 
    #initiate zero shot classification model and summariser
    pipe = pipeline("zero-shot-classification", model = model)
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    word_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    
    summary_list = []
    topic_label = []
    topic_score = []
    #for text in text_list, if the text has length > 1024, chunk up the text, 
    #summarise and finally predict with zero shot classification model
    
    for text in text_list:
        if  len(word_tokenizer.tokenize(text)) > 1024:
            chunks = chunk_string(text, max_words = 800)
            summary = [summarizer(chunk, max_length=514, min_length=min(100, len(word_tokenizer.tokenize(text))), do_sample=False)[0]["summary_text"] for chunk in chunks]
            predict = [pipe(s, candidate_labels, multi_label=True) for s in summary]
        
            score_dict = defaultdict(list)
            
            #each chunk will have a set of labels and their respective scores
            #get the mean scores for each label to represent the text
            for d in predict:
                for label, score in zip(d['labels'], d['scores']):
                    score_dict[label].append(score)
             
            mean_scores = {label: statistics.mean(scores) for label, scores in score_dict.items()}
            topic_label.append(list(mean_scores.keys()))
            topic_score.append(list(mean_scores.values()))
            summary_list.append(summary)

        
        else:
            text = " ".join(text.split())
            summary = summarizer(text, max_length=514, min_length=min(100,len(word_tokenizer.tokenize(text))), do_sample=False)[0]["summary_text"]
            predict = pipe(summary, candidate_labels, multi_label=True)
            topic_label.append(predict['labels'])
            topic_score.append(predict['scores'])
            summary_list.append(summary)
    
    #find the top score and the corresponding label to classify the text
    top_score = [x[x.index(max(x))] for x in topic_score]
    pos = [x.index(max(x)) for x in topic_score]

    top_topic =  [x[idx] for x, idx in zip(topic_label, pos)]
    
    return topic_label, topic_score, top_topic, top_score, summmary