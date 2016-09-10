import requests
import json

# List files and folders in Dropbox
url = "https://api.dropboxapi.com/2/files/list_folder"
headers = {
    "Authorization": "Bearer <This is where the token is to be put>",
    "Content-Type": "application/json"
}
data = {
    "path": ""
}
r = requests.post(url, headers=headers, data=json.dumps(data))

# Download a file from Dropbox
url = "https://content.dropboxapi.com/2/files/download"
headers = {
    "Authorization": "Bearer <This is where the token is to be put>",
    "Dropbox-API-Arg": "{\"path\":\"/Getting Started.pdf\"}"
}
r = requests.post(url, headers=headers)
contents = r.content
with open('Getting Started.pdf', 'wb') as f:
    f.write(contents)

