import pandas as pd
import spacy
from spacy_entity_linker import EntityLinker

def create_NER (doc):
     """
     creates a pandas df with entity, its classification and wiki url

     Args:
         doc (obj): the object returned from nlp(text)
        
     Returns:
         entity_df (pandas df): the pandas df with entity, its classification 
         and wiki url
         
     """ 
  entity_data = []
  
  for ent in doc.ents:
    for linked_ent in doc._.linkedEntities:
      if ent.start == linked_ent.get_span().start:
        entity_data.append({
                "entity": ent.text,
                "entity_label": ent.label_,
                "description": linked_ent.get_description(),
                "wikipedia_url": linked_ent.get_url()
            })
        break  # Assuming one-to-one mapping

  # Convert to DataFrame
  entity_df = pd.DataFrame(entity_data)
  entity_df = entity_df.drop_duplicates()
  #keep only entities that are person or organisation
  entity_df = entity_df[entity_df['entity_label'].isin(["PERSON", "ORG", "NORG"])]

  return entity_df

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("entityLinker", last=True)

df = pd.read_csv('cnbc_news_df.csv')


entity_df = []
#loop through all articles and apply create_NER on each and attached its URL as 
#unique ID for the df
for i in range(len(df)):
  doc = nlp(df['text'][i])
  try:
    temp = create_NER(doc)
    temp['article_url'] =  str(df['redirected_urls'][i])
    entity_df.append(temp)
  except Exception as e:
    pass

final_entity_df = pd.concat(entity_df)

#count the occurrences for each entity
count = final_entity_df.groupby("entity").size().reset_index(name='count')
final_entity_df = pd.merge(final_entity_df, count, how = 'left', on = 'entity')
#replace the entity name for each wiki link with the entity that occurs the most
final_entity_df['entity_cleaned'] = final_entity_df\
    .sort_values(["wikipedia_url",'count'],ascending=[True, False])\
        .groupby("wikipedia_url").transform('first')

final_entity_df.to_csv("data/entity_df.csv", index= False)
