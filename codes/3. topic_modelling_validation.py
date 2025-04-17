import pandas as pd
from classification_helper import predict_topic
from sklearn.metrics import classification_report

#load data
fraud_df = pd.read_csv('data/fraud_df.csv')
tax_evade_df = pd.read_csv('data/tax_evade_df.csv')
positive_df = pd.read_csv('data/positive_df.csv')
neutral_df =  pd.read_csv('data/neutral_df.csv')

#label data with financial crime topic
fraud_df ['topic'] = 'fraud'
tax_evade_df['topic'] = 'tax evasion'
positive_df['topic'] = 'non financial crime'
neutral_df['topic'] = 'non financial crime'

test_df = pd.concat([fraud_df, tax_evade_df, positive_df, neutral_df], ignore_index=True)  

#the categories to classify data into
candidate_labels = ['fraud', 'tax evasion', 'scam', 'other financial crime', 'non financial crime']

#test out the models
test_df['fb_label'], test_df['fb_score'], test_df['top_fb_topic'], test_df['top_fb_score'], summary =\
    predict_topic ("facebook/bart-large-mnli", test_df['text'], candidate_labels)
test_df['ml_label'], test_df['ml_score'], test_df['top_ml_topic'], test_df['top_ml_score'], summary =\
    predict_topic ("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", test_df['text'], candidate_labels)
    
test_df.to_csv('data/topic_modelling_bart_results.csv', index = False)

print(classification_report(test_df['topic'], test_df['top_fb_topic'], zero_division = 0))

# =============================================================================
#                        precision    recall  f1-score   support
# 
#                 fraud       0.79      0.79      0.79        28
#   non financial crime       1.00      0.52      0.68        33
# other financial crime       0.00      0.00      0.00         0
#                  scam       0.00      0.00      0.00         0
#           tax evasion       0.89      0.57      0.70        14
# 
#              accuracy                           0.63        75
#             macro avg       0.53      0.37      0.43        75
#          weighted avg       0.90      0.63      0.72        75
# =============================================================================

print(classification_report(test_df['topic'], test_df['top_ml_topic'], zero_division = 0))
# =============================================================================
#                       precision    recall  f1-score   support
# 
#                 fraud       0.79      0.68      0.73        28
#   non financial crime       0.96      0.76      0.85        33
# other financial crime       0.00      0.00      0.00         0
#                  scam       0.00      0.00      0.00         0
#           tax evasion       1.00      0.43      0.60        14
# 
#              accuracy                           0.67        75
#             macro avg       0.55      0.37      0.44        75
#          weighted avg       0.91      0.67      0.76        75
# =============================================================================
