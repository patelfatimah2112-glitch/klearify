from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory or current working directory
backend_dir = Path(__file__).resolve().parent
load_dotenv(backend_dir / '.env')
load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PRIMARY_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "message": "Klearify Backend is running 🚀",
        "model": PRIMARY_MODEL
    })


@app.route('/explain', methods=['POST'])
def explain():
    data = request.json or {}
    concept = data.get('concept')
    interest = data.get('interest')
    language = data.get('language', 'English')
    level = data.get('level', 'a school student')

    if not concept or not interest:
        return jsonify({"error": "Concept and interest are required"}), 400

    prompt = f"""
You are a fun and simple teacher. Explain concepts in this EXACT format:

📌 DEFINITION:
Give a simple one or two line definition of {concept} in plain words. No jargon.

🏏 EXAMPLE:
Explain {concept} using a real life example from {interest}.
Make the comparison very clear — show exactly how {interest} relates to {concept}.
Every part of the example should map to a part of the concept.

📝 KEY POINTS:
Give 3 to 4 short bullet points about {concept}.
Each point should also use a small {interest} example or comparison.
Keep each point to one or two lines maximum.

STRICT RULES:
- Use simple everyday words only
- No complex technical terms without explanation
- Make it feel like a smart friend explaining on WhatsApp
- Use maximum 3 emojis in the entire explanation. Only use them for the 3 section headings.
- Respond in {language}
- Do NOT change the concept or simplify it wrongly
- The definition must be accurate
- The example must clearly map to the concept

CONCEPT: {concept}
INTEREST: {interest}
LEVEL: {level}
LANGUAGE: {language}
"""

    models_to_try = [PRIMARY_MODEL] + [m for m in FALLBACK_MODELS if m != PRIMARY_MODEL]
    last_error = None

    for model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fun, friendly teacher who explains complex concepts using simple language and relatable examples. Never use complicated words. Always be encouraging."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.7,
                max_tokens=1500
            )
            content = response.choices[0].message.content
            if content:
                return jsonify({"explanation": content})
        except Exception as e:
            print(f"Model {model} failed: {e}")
            last_error = e

    return jsonify({"error": f"Failed to generate explanation: {str(last_error)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
