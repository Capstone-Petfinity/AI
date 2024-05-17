from flask import Flask, render_template
from PIL import Image
import os

app = Flask(__name__)


@app.route("/hello")
def hello():

    return "hello world!"


if __name__ == "__main__":
    app.run()
