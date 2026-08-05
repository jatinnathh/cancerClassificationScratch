from flask import Flask, request, jsonify
from flask_cors import CORS
from gemini_service import get_ai_feedback

app = Flask(__name__)
CORS(app)

@app.route("/gemini-feedback", methods=["POST"])
def gemini_feedback():

    data = request.json
    symptoms = data.get("symptoms")

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    try:
        result = get_ai_feedback(symptoms)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=True)
