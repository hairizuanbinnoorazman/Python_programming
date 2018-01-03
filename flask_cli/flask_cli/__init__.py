from flask import Flask

app = Flask(__name__)

import db
import main
import views