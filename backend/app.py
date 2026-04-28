# Brain of Learnify app
# Takes concept + interest from user
# Uses Gemini AI to explain it in a personalized way

from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load secret API key
load_dotenv()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Connect to Gemini AI
genai.configure(api_key=os.getenv("AIzaSyCaoKRLSiwRtmbL1i0eSF9e8wO6Wcb02rs"))
model = genai.GenerativeModel("gemini-2.0-flash")

# Main route
@app.route('/explain', methods=['POST'])
def explain():
    data = request.json
    concept = data.get('concept')
    interest = data.get('interest')
    language = data.get('language', 'English')

    # Prompt sent to Gemini
    prompt = f"""
    You are a fun and creative teacher.
    Explain the concept of "{concept}" to a student
    using examples related to "{interest}".
    Make it simple, fun, and easy to remember.
    Explain the concept n way as if you are xeplaining it to a 5 year old child.
    Respond in {language}.
    """

    response = model.generate_content(prompt)

    return jsonify({"explanation": response.text})

if __name__ == '__main__':
    app.run(debug=True)