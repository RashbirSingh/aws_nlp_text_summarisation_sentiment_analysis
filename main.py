import pandas as pd
from flask import Flask, render_template
from flask import request, session
import boto3
from boto3.dynamodb.conditions import Key
from newsapi.newsapi_client import NewsApiClient
import en_core_web_sm
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import pdfplumber
import os
from dotenv import dotenv_values
import re
import uuid
from textblob import TextBlob
from matplotlib.figure import Figure
import time
import os
import tweepy as tw
import json
import wget
from pathlib import Path

#python -m spacy download en_core_web_sm
nlp = en_core_web_sm.load()

app = Flask(__name__)
config = dotenv_values(".env")
app.secret_key = "hello"
os.environ["aws_access_key_id"] = config["aws_access_key_id"]
os.environ["aws_secret_access_key"] = config["aws_secret_access_key"]
os.environ["region_name"] = config["region_name"]
newsapi = NewsApiClient(api_key=config['api_key'])
consumer_key= config['consumer_key']
consumer_secret= config['consumer_secret']
access_token= config['access_token']
access_token_secret= config['access_token_secret']
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

dynamodb = boto3.resource('dynamodb', aws_access_key_id=os.getenv("aws_access_key_id"),
                          aws_secret_access_key=os.getenv("aws_secret_access_key"),
                          region_name=os.getenv("region_name"))
athenaclient = boto3.client('athena',  aws_access_key_id=os.getenv("aws_access_key_id"),
                          aws_secret_access_key=os.getenv("aws_secret_access_key"),
                          region_name=os.getenv("region_name"))
s3 = boto3.client('s3', aws_access_key_id=os.getenv("aws_access_key_id"),
                          aws_secret_access_key=os.getenv("aws_secret_access_key"),
                          region_name=os.getenv("region_name"))

def upload_file(file_name, bucket, finalpath):

    s3_client = boto3.client('s3', aws_access_key_id=os.getenv("aws_access_key_id"),
                          aws_secret_access_key=os.getenv("aws_secret_access_key"))
    s3_client.upload_file(file_name, bucket, finalpath)


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

@app.route('/')
def root():
    if 'CurrentActiveUser' in session:
        return render_template('index.html',
                               userlog = "logout",
                               userlogimage = "log-out",
                               userlogtext = " Logout")
    else:
        return render_template('index.html',
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop("CurrentActiveUser", None)
    session.pop("CurrentActiveUserName", None)
    return render_template('login.html')

@app.route("/documentPage", methods=['GET', 'POST'])
def documentPage():
    if 'CurrentActiveUser' in session:
        return render_template('processDocument.html',
                               userlog = "logout",
                               userlogimage = "log-out",
                               userlogtext = " Logout")
    else:
        return render_template('login.html',
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")


@app.route("/processAPI", methods=['GET', 'POST'])
def processAPI():
    if 'CurrentActiveUser' in session:
        return render_template('processAPI.html',
                               userlog = "logout",
                               userlogimage = "log-out",
                               userlogtext = " Logout")
    else:
        return render_template('login.html',
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == "POST":
        req = request.form
        email = str.lower(req.get("email"))
        password = req.get("password")

        table = dynamodb.Table('login')

        response = table.query(
            KeyConditionExpression=Key('email').eq(email)
        )

        if len(response["Items"]) < 1:
            return render_template('notification.html',
                                   notification = "Email or Password is Invalid",
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login"
                                   )
        elif response['Items'][0]['password'] != password:
            return render_template('notification.html',
                                   notification = "Email or Password is Invalid",
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login"
                                   )
        else:
            session["CurrentActiveUser"] = email
            session["CurrentActiveUserName"] = response['Items'][0]['user_name']


        return render_template('index.html',
                               userlog="logout",
                               userlogimage="log-out",
                               userlogtext=" Logout")

    else:
        if 'CurrentActiveUser' in session:
            return render_template('login.html',
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")
        else:
            return render_template('login.html',
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login")



@app.route('/register', methods=['GET', 'POST'])
def register():
    table = dynamodb.Table('login')
    if request.method == "POST":
        req = request.form

        #Getting Form data
        email = str.lower(req.get("email"))
        password = str(req.get("password"))
        user_name = str.lower(req.get("user_name"))


        response = table.query(
            KeyConditionExpression=Key('email').eq(email)
        )

        if len(response["Items"]) < 1:
            table.put_item(
                Item={
                    'email': email,
                    'user_name': user_name,
                    'password': password,
                }
            )
            return render_template('login.html')

        else:
            return render_template('notification.html',
                                   notification="The email already exists",
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login"
                                   )

    if 'CurrentActiveUser' in session:
        return render_template('register.html',
                               userlog = "logout",
                               userlogimage = "log-out",
                               userlogtext = " Logout")
    else:
        return render_template('register.html',
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")



@app.route("/documentsProcess", methods=['GET', 'POST'])
def documentsProcess():
    email = session["CurrentActiveUser"]
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save('static/doc/'+uploaded_file.filename)

        with pdfplumber.open('static/doc/'+uploaded_file.filename) as pdf:
            first_page = pdf.pages[0]
            summary = textSummarizer(first_page.extract_text())
            sentimentScore = textSentimentAnalysis(summary)

        table = dynamodb.Table('usersummary')

        table.put_item(
            Item={
                'count': my_random_string(),
                'email': email,
                'isDocument': 1,
                'for' : uploaded_file.filename,
                'summary': summary,
                'sentimentScore': int(sentimentScore * 100),
            }
        )

        upload_file('static/doc/'+uploaded_file.filename, "storage-s3810585", uploaded_file.filename)

        if 'CurrentActiveUser' in session:
            return render_template('analysisresult.html',
                                   summaryresultdata=summary,
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")
        else:
            return render_template('analysisresult.html',
                                   summaryresultdata=summary,
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login")


@app.route("/APITextToSearch", methods=['GET', 'POST'])
def APITextToSearch():
    APITextToSearch = request.form["TextToSearch"]
    raw = newsTextSummarizer(APITextToSearch)
    if len(raw) < 3:
        if 'CurrentActiveUser' in session:
            return render_template('analysisresult.html',
                                   summaryresultdata="No Results!",
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")
        else:
            return render_template('analysisresult.html',
                                   summaryresultdata="No Results!",
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login")
    summary = textSummarizer(raw)
    sentimentScore = textSentimentAnalysis(summary)

    email = session["CurrentActiveUser"]

    table = dynamodb.Table('usersummary')
    table.put_item(
        Item={
            'count': my_random_string(),
            'email': email,
            'isDocument': 0,
            'for': APITextToSearch,
            'summary': summary,
            'sentimentScore': int(sentimentScore*100),
        }
    )

    while not os.path.exists("static/plots/sentimentgraph.png"):
        time.sleep(1)

    if 'CurrentActiveUser' in session:
        return render_template('analysisresult.html',
                               summaryresultdata = summary,
                               sentimentgraph = "sentimentgraph.png",
                               userlog="logout",
                               userlogimage="log-out",
                               userlogtext=" Logout")
    else:
        return render_template('analysisresult.html',
                               summaryresultdata=summary,
                               sentimentgraph="sentimentgraph.png",
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")


@app.route("/summaryresult", methods=['GET', 'POST'])
def summaryresult():
    email = session["CurrentActiveUser"]
    if 'CurrentActiveUser' in session:
        table = dynamodb.Table('usersummary')
        data = pd.DataFrame(table.scan()['Items'])
        if 'email' in data.columns:
            selecteduser = data.loc[data.email == email, ['for', 'summary', 'isDocument', 'sentimentScore']]
            return render_template('summaryresult.html',
                                   resultvalue = selecteduser.T.to_dict().values(),
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")
        else:
            return render_template('summaryresult.html',
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")
    else:
        return render_template('index.html',
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")

def textSummarizer(text):
    doc = nlp(text)
    corpus = [sent.text.lower() for sent in doc.sents]
    cv = CountVectorizer(stop_words=list(STOP_WORDS))
    cv_fit = cv.fit_transform(corpus)
    word_list = cv.get_feature_names();
    count_list = cv_fit.toarray().sum(axis=0)
    word_frequency = dict(zip(word_list, count_list))
    val = sorted(word_frequency.values())
    higher_word_frequencies = [word for word, freq in word_frequency.items() if freq in val[-3:]]
    # gets relative frequency of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():
        word_frequency[word] = (word_frequency[word] / higher_frequency)

    sentence_rank = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_frequency.keys():
                if sent in sentence_rank.keys():
                    sentence_rank[sent] += word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent] = word_frequency[word.text.lower()]
    top_sentences = (sorted(sentence_rank.values())[::-1])
    top_sent = top_sentences[:8]

    summary = []
    for sent, strength in sentence_rank.items():
        if strength in top_sent:
            summary.append(sent)
        else:
            continue

    finalsummary = [str(i) for i in summary]
    return " ".join(finalsummary)


def newsTextSummarizer(keyword):
    all_articles = newsapi.get_everything(q=keyword,
                                          language='en',)
    endcollection = ["".join(eacharticle["description"].split(".")[:-1]) for eacharticle in all_articles['articles']]
    finalcollection = "".join(endcollection)
    finalcollection =  re.sub(r'\.+', ".", finalcollection)
    finalcollection = re.sub(r'\.\.\.', ".", finalcollection)
    return finalcollection


def textSentimentAnalysis(text):
    my_file = Path("static/plots/sentimentgraph.png")
    if my_file.is_file():
        os.remove("static/plots/sentimentgraph.png")
    blob = TextBlob(text)
    for sentence in blob.sentences:
        summarypolarity = sentence.sentiment.polarity

    # Pie chart
    labels = ['', 'Sentiment polarity']
    sizes = [100-abs(summarypolarity*100), abs(summarypolarity*100)]
    explode = (0, 0.1)
    if summarypolarity >= 0:
        colors = ['silver', 'green']
    else:
        colors = ['silver', 'red']

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    fig.savefig("static/plots/sentimentgraph.png")

    return summarypolarity

@app.route("/ProcessTwitterAPI", methods=['GET', 'POST'])
def ProcessTwitterAPI():
    if 'CurrentActiveUser' in session:
        return render_template('processTwitterAPI.html',
                               userlog = "logout",
                               userlogimage = "log-out",
                               userlogtext = " Logout")
    else:
        return render_template('login.html',
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")


def athenaScript():
    query = 'SELECT * FROM "twitterdatabase"."twitterdata" limit 10;'
    database = "twitterdatabase"
    athena_result_bucket = "s3://storage-s3810585/athenaoutput/"

    response = athenaclient.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': athena_result_bucket,
        }
    )
    return response["QueryExecutionId"]

def pushTweetTos3(keyword):
    tweets = tw.Cursor(api.search,
                       q=keyword,
                       lang="en").items(10)

    # Iterate and print tweets
    i = 0
    for tweet in tweets:
        i = i + 1
        with open("static/tweet/Tweet_"+str(i)+".json", 'w') as f:
            json.dump(tweet._json, f)
            f.close()
            upload_file("static/tweet/Tweet_"+str(i)+".json", "storage-s3810585",
                        "TwitterData/data/Tweet_"+str(i)+".json")


@app.route("/TwitterTextToSearch", methods=['GET', 'POST'])
def TwitterTextToSearch():
    APITextToSearch = request.form["TextToSearch"]
    pushTweetTos3(APITextToSearch)
    processedDataName = athenaScript()
    time.sleep(3)
    s3.download_file("storage-s3810585", "athenaoutput/"+processedDataName+".csv", "static/athenacsv/"+processedDataName+".csv")

    df = pd.read_csv("static/athenacsv/"+processedDataName+".csv")
    raw = ".".join(df.text)
    if len(raw) < 3:
        if 'CurrentActiveUser' in session:
            return render_template('analysisresult.html',
                                   summaryresultdata="No Results!",
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")
        else:
            return render_template('analysisresult.html',
                                   summaryresultdata="No Results!",
                                   userlog="login",
                                   userlogimage="log-in",
                                   userlogtext=" Login")
    summary = textSummarizer(raw)
    sentimentScore = textSentimentAnalysis(summary)

    email = session["CurrentActiveUser"]

    table = dynamodb.Table('usersummary')
    table.put_item(
        Item={
            'count': my_random_string(),
            'email': email,
            'isDocument': 0,
            'for': APITextToSearch,
            'summary': summary,
            'sentimentScore': int(sentimentScore*100),
        }
    )

    while not os.path.exists("static/plots/sentimentgraph.png"):
        time.sleep(1)

    if 'CurrentActiveUser' in session:
        return render_template('analysisresult.html',
                               summaryresultdata = summary,
                               sentimentgraph = "sentimentgraph.png",
                               userlog="logout",
                               userlogimage="log-out",
                               userlogtext=" Logout")
    else:
        return render_template('analysisresult.html',
                               summaryresultdata=summary,
                               sentimentgraph="sentimentgraph.png",
                               userlog="login",
                               userlogimage="log-in",
                               userlogtext=" Login")

    return render_template('index.html',
                                   summaryresultdata="No Results!",
                                   userlog="logout",
                                   userlogimage="log-out",
                                   userlogtext=" Logout")





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)


