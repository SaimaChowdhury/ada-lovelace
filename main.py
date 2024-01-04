import pandas as pd
from flask import Flask, render_template, request
from numpy import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from chatterbot import ChatBot
from chatterbot.conversation import Statement
from chatterbot.trainers import ChatterBotCorpusTrainer

import time  # for solving AttributeError: module 'time' has no attribute 'clock'

time.clock = time.time  # for solving AttributeError: module 'time' has no attribute 'clock'

#####################################################

app = Flask(__name__)

#####################################################
# Loading Dataset Globally
data = pd.read_csv("ada_dataset_1000.csv")

le_gender = LabelEncoder()
le_age = LabelEncoder()
le_personality = LabelEncoder()

data['gender_n'] = le_gender.fit_transform(data['Gender'])
data['age_n'] = le_age.fit_transform(data['Age'])
data['personality_n'] = le_personality.fit_transform(data['Personality Type'])

print(data.head(10))

X = data[['gender_n', 'age_n', 'Talkative', 'Start Conversations', 'Interested in People', 'Make People Feel at Ease',
          'concern for others', 'Full of Ideas', 'Always Prepared', 'Get Irritated Easily', 'Always Ready to Sleep',
          'Frequent Mood Swings',
          'Seek Adventure', 'Like to Be in Charge', 'Make a Mess of Things', 'Leave Belongings Around',
          'Forget to Put Things Back', 'Follow a Schedule', 'Overcome Failure',
          "Leave Today's Work for Next Day", 'Difficulty Understanding Abstract Ideas', 'Regret After Making Decisions',
          'Prefer Daydreaming to Reality', 'Like to Express Immediate Mental/Emotional Situations',
          'Prefer to Build New Relationships Rather Than Compromise', 'Think No Rules in Society Would Be Better',
          "Don't Think This Test Can Determine Personality"]]

y = data['personality_n']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=.30, random_state=1)
model = LogisticRegression(max_iter=1000)
model.fit(Xtrain.values, ytrain.values)  # .values - for solving error X does not have valid feature names

# For chatbot
chatbot = ChatBot('ChatBot')
trainer = ChatterBotCorpusTrainer(chatbot)
trainer.train("chatterbot.corpus.english")


#####################################################


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/test", methods=["POST", "GET"])
def test():
    if request.method == "GET":
        return render_template("test.html")

    else:
        age = int(request.form["age"])
        if age < 17:
            age = 17
        elif age > 28:
            age = 28

        gender = str(request.form["gender"])
        if (gender == "Female"):
            gender = 0
        elif (gender == "Male"):
            gender = 1

        inputdata = [
            [
                gender,
                age,
                9 - int(request.form["qsn001"]),
                9 - int(request.form["qsn002"]),
                9 - int(request.form["qsn003"]),
                9 - int(request.form["qsn004"]),
                9 - int(request.form["qsn005"]),

                9 - int(request.form["qsn006"]),
                9 - int(request.form["qsn007"]),
                9 - int(request.form["qsn008"]),
                9 - int(request.form["qsn009"]),
                9 - int(request.form["qsn010"]),

                9 - int(request.form["qsn011"]),
                9 - int(request.form["qsn012"]),
                9 - int(request.form["qsn013"]),
                9 - int(request.form["qsn014"]),
                9 - int(request.form["qsn015"]),

                9 - int(request.form["qsn016"]),
                9 - int(request.form["qsn017"]),
                9 - int(request.form["qsn018"]),
                9 - int(request.form["qsn019"]),
                9 - int(request.form["qsn020"]),

                9 - int(request.form["qsn021"]),
                9 - int(request.form["qsn022"]),
                9 - int(request.form["qsn023"]),
                9 - int(request.form["qsn024"]),
                9 - int(request.form["qsn025"]),
            ]
        ]

        dt = pd.DataFrame(inputdata)
        testdf = dt[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]]
        testData = testdf.values

        global per #declare per as a global variable

        per = model.predict(testData)
        if (per == 0):
            per = "Agreeableness"
        elif (per == 1):
            per = "Conscientinousness"
        elif (per == 2):
            per = "Extraversion"
        elif (per == 3):
            per = "Neuroticism"
        else:
            per = "Openness"

        return render_template("result.html", per=per)


@app.route("/result_details")
def result_details():
    if (per == "Agreeableness"):
        return render_template("type4-ag.html")
    elif (per == "Conscientinousness"):
        return render_template("type2-cns.html")
    elif (per == "Extraversion"):
        return render_template("type3-ex.html")
    elif (per == "Neuroticism"):
        return render_template("type5-nu.html")
    else:
        return render_template("type1-op.html")


@app.route("/types")
def types():
    return render_template("types.html")


@app.route("/type1")
def type1():
    return render_template("type1-op.html")


@app.route("/type2")
def type2():
    return render_template("type2-cns.html")


@app.route("/type3")
def type3():
    return render_template("type3-ex.html")


@app.route("/type4")
def type4():
    return render_template("type4-ag.html")


@app.route("/type5")
def type5():
    return render_template("type5-nu.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/faq")
def faq():
    return render_template("FAQ.html")


@app.route("/contacts")
def contacts():
    return render_template("contacts.html")


@app.route("/process")
def process():
    return render_template("process.html")


@app.route("/map")
def map():
    return render_template("map.html")


# Handling error 404
@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", code=404, text="Page Not Found"), 404


# Handling error 500
@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", code=500, text="Internal Server Error"), 500


# Chatbot
@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')  # use request.args in a Flask route to access the value of a URL parameter ;
    # Get the value of the 'msg' parameter from the URL
    return str(chatbot.get_response(Statement(text=userText, search_text=userText)))


if __name__ == "__main__":
    app.run()
