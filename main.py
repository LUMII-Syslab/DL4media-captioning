import os, json
from flask import Flask, request

from captions import generate_caption

app = Flask(__name__)
app.config["SECRET_KEY"] = "model!"


@app.route('/')
def get_index():
	return "Server for model is running"


@app.route('/predict-caption', methods=["POST"])
def predict_caption():
	files = request.files.getlist("file")
	file = files[0]

	result = generate_caption(file)
	
	return result


if __name__ == '__main__':
	app.run(debug=False, host='0.0.0.0', threaded=True)
