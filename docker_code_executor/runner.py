import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Union

import docker

client = docker.DockerClient(base_url="unix:///var/run/docker.sock")

def get_db():
    con = sqlite3.connect("example.db")
    return con 
 
def get_job() -> Union[dict, None]:
    con = get_db()
    cur = con.cursor()
    vals = cur.execute('SELECT * from job where status = "submitted"')
    items = vals.fetchone()
    if items is None:
        return None
    return {
        "id": items[0],
        "runtime": items[1],
    }

def update_job(id: str, status: str):
    con = get_db()
    cur = con.cursor()
    current_time = datetime.now()
    cur.execute("UPDATE job SET status = ?, date_modified = ? WHERE id = ?", (status, current_time, id))
    con.commit()

def get_language_mapping(runtime: str, folder_name: str) -> Union[str, None]:
    just_lang = runtime.split(":")[0]
    language_mapping = {
        "python": f"python /{folder_name}/main.py",
        "golang": f"go run /{folder_name}/main.go"
    }

    return language_mapping.get(just_lang)

    

def run_script(image_name: str, folder_name: str):
    print("begin pull")
    client.images.pull(image_name)     
    script_path = os.path.join(os.getcwd(), folder_name)
    log_path = os.path.join(script_path, "output.log")
    print(f"start application :: {script_path} :: {image_name}")

    hh = client.containers.run(
        image=image_name,
        command=get_language_mapping(image_name, folder_name), 
        detach=True,
        read_only=False,
        stdout=True,
        stderr=True,
        # Limit resources
        pids_limit=10,
        mem_limit='200m',
        cap_drop=["ALL"],
        network_mode="none",
        volumes={f"{script_path}": {"bind": f"/{folder_name}", "mode": "ro"}}
    )
    print(f"Container running {hh.name} - {hh.short_id} - {hh.status}")

    current_time = datetime.now()
    max_time = current_time + timedelta(seconds=60)
    completed = False
    while (hh.status == "running" or hh.status == "created") and current_time <= max_time:
        print(f"command is incomplete - {hh.status} - {max_time} - {current_time}")
        current_time = datetime.now()
        hh.reload()
        print(hh.status)
        if hh.status == "exited":
            completed = True
            break
        time.sleep(1)

    output = hh.logs()
    with open(log_path, 'wb+') as f:
        f.write(output)

    if not completed:
        print("command is still incomplete. forcibly kill container")
        hh.kill()
        time.sleep(5)
        hh.remove(force=True)
    else:
        print("command completed. will remove container")
        hh.remove(force=True)


while True:
    aa = get_job()
    if aa is None:
        print("no job so far")
        time.sleep(5)
        continue

    print(f"update job {aa['id']} as running")
    update_job(aa['id'], "running")
    try:
        run_script(aa['runtime'], aa['id'])
        print(f"update job {aa['id']} as complete")
        update_job(aa['id'], "complete")
    except Exception as e:
        print(f"error happened: {e}")
        print(f"update job {aa['id']} as failed")
        update_job(aa['id'], "failed")
    
    time.sleep(5)
