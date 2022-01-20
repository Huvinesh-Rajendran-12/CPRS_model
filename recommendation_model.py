import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

client_data = pd.read_csv('./data/clients.csv',skipinitialspace = True)

student_group_data = pd.read_csv('./data/group.csv',skipinitialspace =True)

#prepare client data 
client_data['Requirements'] = client_data['Project brief '].map(str) + ' ' + client_data['Project Background']+' '+client_data['Requiements']

client_data = client_data.drop(['Requiements'],axis=1)

client_data = client_data.fillna(' ')

stopwords_ = stopwords.words('english')
stopword_ = set(stopwords.words('english'))
wn = WordNetLemmatizer()

# Create word tokens
def token_txt(token):
    return token not in stopword_ and token not in list(string.punctuation) and len(token) > 2

def clean_txt(text):
  clean_text = []
  clean_text2 = []
  text = re.sub("'", "", text)
  text = re.sub("(\\d|\\W)+", " ", text) 
  text = text.replace("nbsp", "")
  clean_text = [wn.lemmatize(word, pos = "v") for word in word_tokenize(text.lower()) if token_txt(word)]
  clean_text2 = [word for word in clean_text if token_txt(word)]
  return " ".join(clean_text2)

client_data['Requirements'] = client_data['Requirements'].apply(clean_txt)

#prepare student group corpus 
student_group_data['group_details '] = student_group_data['group_details '].apply(clean_txt)

#term frequency - inverse document frequency 
tfidf_vect = TfidfVectorizer()

# Fitting and transforming the vector
tfidf_comb = tfidf_vect.fit_transform((client_data['Requirements'])) 
tfidf_comb 

#input user group id 
group_id = int(input("Enter the group id: "))

grp_index = np.where(student_group_data['group_id'] == group_id)[0][0]

print(grp_index)

student_group_data = student_group_data.iloc[[grp_index]]

print(student_group_data)

stud_group_rec = tfidf_vect.transform(student_group_data['group_details '])
cos_stud_group_rec = list(map(lambda x: cosine_similarity(stud_group_rec, x), tfidf_comb))

def get_recommendation(user,top, client_data, scores):
  recommendation = pd.DataFrame(columns = ['Group', 'Title',  'Company Name', 'Score'])
  count = 0
  for i in top:
      recommendation.at[count, 'Group'] = user
      recommendation.at[count, 'Title'] = client_data['Title'][i]
      recommendation.at[count, 'Company Name'] = client_data['Company Name'][i]
      recommendation.at[count, 'Score'] =  scores[count]
      count += 1
  return recommendation

top10_stud_group_rec = sorted(range(len(cos_stud_group_rec)), key = lambda i: cos_stud_group_rec[i], reverse = True)[:10]

list_scores = [cos_stud_group_rec[i][0][0] for i in top10_stud_group_rec] 

print(get_recommendation(group_id,top10_stud_group_rec, client_data, list_scores))





