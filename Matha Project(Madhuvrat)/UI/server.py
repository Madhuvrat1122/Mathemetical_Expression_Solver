from flask import Flask, request, render_template, url_for
from test import solve
app = Flask(__name__)
@app.route("/main")
def home():
    return render_template("hand.html")

@app.route("/result",methods=["POST"])
def output():
    form_data = request.form
    status = solve(form_data["fileupload"])
    return render_template("hand.html",status=status[0],status1=status[1])

if __name__ == "__main__":
    app.run(debug=True)