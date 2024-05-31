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

def get_job(id: str) -> dict:
    con = get_db()
    cur = con.cursor()
    print(id)
    vals = cur.execute("SELECT * from job WHERE id = ?", [id])
    items = vals.fetchone()
    if items is None:
        return ''
    return {
        "id": items[0],
        "runtime": items[1],
        "status": items[2], 
    }
 
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
    runtime = incoming_input["runtime"]
    decoded_code = base64.b64decode(raw_code)

    key = str(uuid.uuid4())
    status = "submitted"
    current_time = datetime.now()

    # Create folders
    script_path = os.path.join(os.getcwd(), key)
    print(f"creating directory {script_path}")
    os.mkdir(script_path)

    # write out code file
    code_path = get_code_path(script_path, runtime)
    print(f"Saving code {code_path}")
    with open(code_path, "wb+") as f:
        f.write(decoded_code)

    con = get_db()
    cur = con.cursor()
    cur.execute("INSERT INTO job VALUES (?, ?, ?, ?, ?)",  (key,runtime,status,current_time,current_time))
    con.commit()
    return {} 

def get_code_path(script_path: str, runtime: str) -> str:
    code_path = ""
    short_lang = runtime.split(":")[0]
    if short_lang == "python":
        code_path = os.path.join(script_path, "main.py")
    elif short_lang == "golang": 
        code_path = os.path.join(script_path, "main.go")

    return code_path


@app.route("/view-submission/<id>")
def view_submission(id):
    script_path = os.path.join(os.getcwd(), id)
    log_path = os.path.join(script_path, "output.log")
    raw_code = ''
    raw_log = ''

    job_info = get_job(id)
    code_path = get_code_path(script_path, job_info["runtime"])

    with open(code_path, 'r') as f:
        raw_code = f.read()

    if job_info["status"] == "complete":
        with open(log_path, 'r') as f:
            raw_log = f.read()
    else:
        raw_log = f"no log available - status of job: {job_info['status']}"

    raw_code_split = raw_code.split('\n')
    raw_log_split = raw_log.split('\n')

    return render_template("single_submission.html", id=id, status=job_info["status"],  multiline_code=raw_code_split, multiline_log=raw_log_split, runtime=job_info["runtime"])


@app.route("/")
def hello_world():
    return render_template("index.html")


@app.route("/create-submission")
def ui_create_submission():
    return render_template("create_submission.html")

