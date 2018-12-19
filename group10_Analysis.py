### ===Imports===
import pandas as pd
import sshtunnel
import numpy as np
import MySQLdb
import datetime
import scipy
from nltk.corpus import stopwords
from nltk.corpus import twitter_samples
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import nltk.classify
import math
import pickle
import string

### ===Querying MySQL db via PythonAnywhere, storing results in DataFrame===
sshtunnel.SSH_TIMEOUT = 5.0
sshtunnel.TUNNEL_TIMEOUT = 5.0

with sshtunnel.SSHTunnelForwarder(
    ('ssh.pythonanywhere.com'),
    ssh_username='your_account', ssh_password='your_password',
    remote_bind_address=('your_account.mysql.pythonanywhere-services.com', 3306)
) as tunnel:
    connection = MySQLdb.connect(
        user='your_account', password='your_password',
        host='127.0.0.1', port=tunnel.local_bind_port,
        database='your_account$default',
    )
    c = connection.cursor()
    c.execute("SELECT * FROM tweets")
    connection.close()

rows = c.fetchall()
twid = []
content = []
date = []
usid = []
followers = []
retweet = []
for eachRow in rows:

    twid.append(eachRow[0])
    content.append(eachRow[1])
    date.append(eachRow[2])
    usid.append(eachRow[3])
    followers.append(eachRow[4])
    retweet.append(eachRow[5])

d = {'tweet_id':twid, 'tweet':content, 'date':date, 'user_id':usid, 'follower_count':followers, 'retweet_count':retweet}
df = pd.DataFrame(d)

df.drop(df.index[0], inplace=True)

df['date'] = pd.to_datetime(df['date'])

cities = ["Atlanta", "Austin", "Boston", "Chicago", "Columbus", "Dallas", "Denver", "Indianapolis", "Los Angeles",
"Miami", " Montgomery", "Nashville", "Newark", "New York", 'nyc', "Virginia", "Philadelphia", "Pittsburgh", "Raleigh",
"Toronto", "dc", "D.C.", "Arlington"]

def city(text):
    for city in cities:
        locations = []
        if city.lower() in text.lower():
            locations.append(city)
            return locations[0]

df['location'] = df['tweet'].apply(city)

def alt(text):
    locations = []
    for city in cities:
        if city.lower() in text.lower():
            locations.append(city)
            if len(locations)>1:
                return locations[1]

df['alt_location'] = df['tweet'].apply(alt)


def rt(text):
    if 'RT' in text:
        return True
    else:
        return False

df['is_retweet'] = df['tweet'].apply(rt)

df[['follower_count', 'retweet_count']] = df[['follower_count', 'retweet_count']].astype(int)

df.sort_values(['date'],ascending=False, inplace=True)

###===Total tweets per city===
print('Primary city mentions: ')
print(df['location'].value_counts(),'\n')
print('Secondary city mentions: ')
print(df['alt_location'].value_counts(),'\n')


###===Tweet frequency plot===
df['date'].value_counts().plot()

###===Import corpus for training data===
corpus = pd.read_csv('tweetCorpus.txt', sep='\t')
corpus.drop('Unnamed: 2', axis=1, inplace=True)

###===Function to pre-process Tweets===
def clean_text(text):

    """Remove punctuation, lower_case all letters, and remove stopwords.
    Then feed result to algorithm."""

    no_sym = []
    for c in text:
        if c not in string.punctuation:
            no_sym.append(c)

    no_sym = ''.join(no_sym)

    content = []
    for w in no_sym.split():
        if w.lower() not in stopwords.words('english'):
            content.append(w.lower())
    return content

###===Training classification model===
# Use random state=108 to achieve same results.

bow = CountVectorizer(analyzer=clean_text).fit(corpus['text'])

tweet_bow = bow.transform(corpus['text'])

tfidf = TfidfTransformer().fit(tweet_bow)

tweet_tfidf = tfidf.transform(tweet_bow)

model = LogisticRegression().fit(tweet_tfidf, corpus['text'])

text_train, text_test, value_train, value_test = train_test_split(corpus['text'], corpus['value'], test_size=0.3,
                                                random_state=108)

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=clean_text)),
    ('tfidf', TfidfTransformer()),
    ('model', LogisticRegression()),
])


pipeline.fit(text_train,value_train)

predictions = pipeline.predict(text_test)

print(classification_report(predictions,value_test))

###===Save classifier for later use===
# Pickling our classifier allows for later use without retraining the algorithm
save_classifier = open("sentclass1.pickle", "wb")
pickle.dump(pipeline, save_classifier)
save_classifier.close()

classifier_f = open('sentclass1.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

###===Discard bias tweets (following official decision of hq2)===
# Amazon hq2 location announced November 13th. Tweets after this date are highly subject to bias and should not be
    # used for sentiment analysis.
d1 = datetime.datetime(2018, 11, 13)
t = df[df['date']<d1]

###===Classify tweets===
t['sentiment'] = classifier.predict(t['tweet'])

###===Population sizes relevant===
# https://en.wikipedia.org/wiki/List_of_United_States_cities_by_population
# https://en.wikipedia.org/wiki/Toronto
pops = {"Atlanta":486290, "Austin":950715, "Boston":685094, "Chicago":2716450, "Columbus":879170,
"Dallas":1341075, "Denver":704621, "Indianapolis":863002, "Los Angeles":3999759, "Miami":463347,
"Nashville":667560, "Newark":285154, "New York":8622698, "Philadelphia":1580863, "Pittsburgh":302407,
"Raleigh":464758, "Toronto":2731571, "dc":693972, "Arlington":207627}

###===Compute sent score===
def score(city):
    loc = t[t['location']==city]
    if len(loc) > 0:

        ratio = len(loc[loc['sentiment']==1])/len(loc)

        circ = sum(loc['follower_count'])
        pop = pops[city]
        r_circ = circ/pop

        num = ratio * r_circ * 100
        return num
    else:
        pass

###===Organize cities and their sent scores in a dictionary===
    rank = {}
for city in pops.keys():
    try:
        if score(city) > 0:
            rank[city] = score(city)
    except:
        TypeError

###===Open zillow data as DataFrame===
v = pd.read_csv('medianValuesUpdated2.tsv',sep='\t')
v['Month'] = pd.to_datetime(v['Month'])
import datetime
end17 = datetime.datetime(2017, 12, 31)
obsv = v[v['Month']>end17]
trng = v[v['Month']<end17]

###===Define function to determine unexpected growth===
def growth(city):
    y = np.asarray(trng[city])
    x = np.asarray(trng['Month'].map(datetime.datetime.toordinal).values.reshape(-1,1))
    model = linear_model.LinearRegression()
    model.fit(x,y)
    x_test = np.asarray(obsv['Month'].map(datetime.datetime.toordinal).values.reshape(-1,1))
    y_test = AtlantaModel.predict(x_test)
    y_actual = np.asarray(obsv[city])
    p_growth = (y_test[-1] - y_test[0])/y_test[0]*100
    a_growth = (y_actual[-1] - y_actual[0])/y_actual[0]*100
    return a_growth - p_growth

###===Organize cities and unexpected growth in a dictionary===
g = {}
for i in v.columns[2:]:
    g[i] = growth(i)

###===Compare sentiment score to unexpected growth===
sent_x = []
price_y = []
for i in rank.keys():
    for j in g.keys():
        if i.lower() == j.lower():
            sent_x.append(rank[i])
            price_y.append(g[j])
r_sq = np.corrcoef(sent_x, price_y)[0,1]
print(r_sq)

###==Compute recommendation score for client===
s = np.array(sent_x)
p = np.array(price_y)
rec = {}
for i,n in zip((s*p).reshape(-1,1), rank.keys()):
        rec[n] = i
# Recommendation to our client based on score. Atlanta is considerably higher than runner-up!
print(rec)
