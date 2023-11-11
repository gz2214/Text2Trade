import json
import urllib.request
import pandas as pd 
from datetime import datetime
import time 
import json
apikey = "your_api_key"
category='business'
dates=pd.date_range(start="2020-09-12",end='2023-11-11')




articles=[]

#%%
#business
count=0
for d in dates:
    day=str(d)[:10]+'T00:00:00Z'
    url =f"https://gnews.io/api/v4/top-headlines?category={category}&lang=en&country=us&max=50&to={day}&sortby=relevance&expand=content&apikey={apikey}"
    print(day)
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        articles =articles+ data["articles"]
    if count>6:
        time.sleep(5) 
        count=0
    else:
        count+=1
#%%
with open('business_news.json', 'w') as file:
    json.dump(articles, file)

#%%
#technology 
category='technology'
count=0
tech_articles=[]

for d in dates:
    day=str(d)[:10]+'T00:00:00Z'
    url =f"https://gnews.io/api/v4/top-headlines?category={category}&lang=en&country=us&max=50&to={day}&sortby=relevance&expand=content&apikey={apikey}"
    print(day)
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode("utf-8"))
        tech_articles =tech_articles+ data["articles"]
    if count>6:
        time.sleep(5) 
        count=0
    else:
        count+=1
#%%
with open('tech_news.json', 'w') as file:
    json.dump(tech_articles, file)
