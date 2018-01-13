# Testing out slack api here
import requests

token = "XXXX"

url = "https://slack.com/api/channels.list"
headers = {"Content-Type": "application/x-www-form-urlencoded"}
data = {"token": token}

response = requests.get(url, headers=headers, params=data)

data = response.content
data2 = response.json()

[{x['id'], x['name']} for x in data2['channels']]


upload_url = "https://slack.com/api/files.upload"
file_name = "test.jpg"
channel_id = "XXXXX"
data = {"token": token,
        "channels": channel_id}
file = {'file': open('./API/test.jpg', 'rb')}

nyaa = requests.post(upload_url, params=data, files=file)

