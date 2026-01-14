from flask import Flask, request, jsonify
from flask import Flask, request, jsonify, send_from_directory
from inference.pipeline import analyze_text

app = Flask(__name__)

@app.route("/", methods=["GET"])
def serve_ui():
    return send_from_directory("../frontend", "index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json["text"]
    return jsonify(analyze_text(text))

if __name__ == "__main__":
    app.run(debug=True)

