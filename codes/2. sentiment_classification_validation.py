import pandas as pd
from classification_helper import predict_sentiment
from sklearn.metrics import classification_report

#load datasets
fraud_df = pd.read_csv('data/fraud_df.csv')
tax_evade_df = pd.read_csv('data/tax_evade_df.csv')
positive_df = pd.read_csv('data/positive_df.csv')
neutral_df =  pd.read_csv('data/neutral_df.csv')

#label their sentiments
fraud_df ['sentiment'] = 'negative'
tax_evade_df['sentiment'] = 'negative'
positive_df['sentiment'] = 'positive' 
neutral_df['sentiment'] = 'neutral'

#concat
test_df = pd.concat([fraud_df, tax_evade_df, positive_df, neutral_df],ignore_index= True)  

#test the various sentiment classification model on the data and get their sentiment label and score
test_df['prosus_label'], test_df['prosus_score'], summary = predict_sentiment ("ProsusAI/finbert", test_df['text'])
test_df['yiyanghkust_label'], test_df['yiyanghkust_score'], summary = predict_sentiment ("yiyanghkust/finbert-tone", test_df['text'])
test_df['mrm8488_label'], test_df['mrm8488_score'], summary = predict_sentiment ("mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", test_df['text'])

test_df['yiyanghkust_label'] = test_df['yiyanghkust_label'].str.lower()

test_df.to_csv('data/sentiment_classification_results.csv', index = False)

print(classification_report(test_df['sentiment'], test_df['prosus_label']))

# =============================================================================
#               precision    recall  f1-score   support
# 
#     negative       0.94      0.76      0.84        42
#      neutral       0.52      0.68      0.59        19
#     positive       0.62      0.71      0.67        14
# 
#     accuracy                           0.73        75
#    macro avg       0.70      0.72      0.70        75
# weighted avg       0.78      0.73      0.75        75
# =============================================================================
print(classification_report(test_df['sentiment'], test_df['yiyanghkust_label']))

# =============================================================================
#               precision    recall  f1-score   support
# 
#     negative       0.95      0.45      0.61        42
#      neutral       0.18      0.26      0.21        19
#     positive       0.41      0.79      0.54        14
# 
#     accuracy                           0.47        75
#    macro avg       0.51      0.50      0.45        75
# weighted avg       0.65      0.47      0.50        75
# =============================================================================
print(classification_report(test_df['sentiment'], test_df['mrm8488_label']))

# =============================================================================
#               precision    recall  f1-score   support
# 
#     negative       0.80      0.19      0.31        42
#      neutral       0.16      0.32      0.21        19
#     positive       0.44      0.86      0.59        14
# 
#     accuracy                           0.35        75
#    macro avg       0.47      0.45      0.37        75
# weighted avg       0.57      0.35      0.33        75
# =============================================================================

print(classification_report(test_df['sentiment'], test_df['fb_label']))
# =============================================================================
#            precision    recall  f1-score   support
# 
#     negative       0.93      0.93      0.93        42
#      neutral       0.00      0.00      0.00        19
#     positive       0.36      0.86      0.51        14
# 
#     accuracy                           0.68        75
#    macro avg       0.43      0.60      0.48        75
# weighted avg       0.59      0.68      0.62        75
# =============================================================================
