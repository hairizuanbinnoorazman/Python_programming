import base64
import os
import sqlite3
import uuid
from datetime import datetime

from flask import Flask, render_template, request

app = Flask(__name__) 

def get_db():
    con = sqlite3.connect("example.db")
    return con
 
@app.route("/submissions", methods=["GET"])
def list_submissions():
    con = get_db()
    cur = con.cursor()
    vals = cur.execute("SELECT * from job")
    items = vals.fetchall()
    return render_template("submission_list.html", submissions=items)

@app.route("/submissions", methods=['POST'])
def create_submission():
    # Get values
    incoming_input = request.json
    raw_code = incoming_input["code"]
    decoded_code = base64.b64decode(raw_code)

    key = str(uuid.uuid4())
    status = "submitted"
    current_time = datetime.now()

    # Create folders
    script_path = os.path.join(os.getcwd(), key)
    print(f"creating direcctory {script_path}")
    os.mkdir(script_path)

    # write out code file
    code_path = os.path.join(script_path, "main.py")
    print(f"Saving code {code_path}")
    with open(code_path, "wb+") as f:
        f.write(decoded_code)

    con = get_db()
    cur = con.cursor()
    cur.execute("INSERT INTO job VALUES (?, ?, ?, ?)",  (key,status,current_time,current_time))
    con.commit()
    return {} 


@app.route("/view-submission/<id>")
def view_submission(id):
    script_path = os.path.join(os.getcwd(), id)
    code_path = os.path.join(script_path, "main.py")
    log_path = os.path.join(script_path, "output.log")
    raw_code = ''
    raw_log = ''
    with open(code_path, 'r') as f:
        raw_code = f.read()

    with open(log_path, 'r') as f:
        raw_log = f.read()

    return render_template("single_submission.html", id=id, code=raw_code, log=raw_log)


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/create-submission")
def ui_create_submission():
    return render_template("create_submission.html")

