from flask_cli import app

@app.route("/")
def lol():
	return 'lol'