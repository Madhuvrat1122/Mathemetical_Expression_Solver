from flask import Flask, render_template, request,jsonify,url_for
from test1 import solve
app = Flask(__name__)

@app.route("/")
def view_template():
    return render_template("index.html")


@app.route("/data", methods=["GET","POST"])
def form_data():
    if request.method == "GET":
        return "<h1>Sorry, You mistaken somewhere</h1>"
    else:
        user_data = request.form   #take data from form in html page
        selected = user_data['selected']
        if int(selected)==0:
            imge = solve(user_data["img_name"])   
            print(imge)
            result=imge[1]
            return jsonify(msg=str(result))
        



if __name__ == "__main__":
  app.run(debug=False,port=8000)