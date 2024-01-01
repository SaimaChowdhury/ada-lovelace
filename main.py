#####################################################
import pandas as pd
from flask import Flask, render_template, request
from numpy import *
from sklearn import linear_model
#####################################################

app = Flask(__name__)

#####################################################
# Loading Dataset Globally
data = pd.read_csv("ada_dataset_1000.csv")
array = data.values

for i in range(len(array)):
    if array[i][0] == "Male":
        array[i][0] = 1
    else:
        array[i][0] = 0

df = pd.DataFrame(array)

maindf = df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26]]
mainarray = maindf.values

temp = df[27]
train_y = temp.values
train_y = temp.values

for i in range(len(train_y)):
    train_y[i] = str(train_y[i])

mul_lr = linear_model.LogisticRegression(
    multi_class="multinomial", solver="newton-cg", max_iter=1000
)
mul_lr.fit(mainarray, train_y)


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
        if age < 16:
            age = 16
        elif age > 29:
            age = 29

        inputdata = [
            [
                request.form["gender"],
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

        for i in range(len(inputdata)):
            if inputdata[i][0] == "Male":
                inputdata[i][0] = 1
            else:
                inputdata[i][0] = 0

        df1 = pd.DataFrame(inputdata)
        testdf = df1[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,19, 20, 21, 22, 23, 24, 25, 26]]
        maintestarray = testdf.values

        y_pred = mul_lr.predict(maintestarray)
        for i in range(len(y_pred)):
            y_pred[i] = str((y_pred[i]))
        DF = pd.DataFrame(y_pred, columns=["Predicted Personality"])
        DF.index = DF.index + 1
        DF.index.names = ["Person No"]

        return render_template(
            "result.html", per=DF["Predicted Personality"].tolist()[0]
        )



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


# Handling error 404
@app.errorhandler(404)
def not_found_error(error):
    return render_template("error.html", code=404, text="Page Not Found"), 404


# Handling error 500
@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", code=500, text="Internal Server Error"), 500











if __name__ == "__main__":
    app.run(debug=True)

if __name__ == "__main__":
    # use 0.0.0.0 for replit hosting
    app.run(host="0.0.0.0", port=8080)

    # for localhost testing
    # app.run()