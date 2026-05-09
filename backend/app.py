from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@app.route('/explain', methods=['POST'])
def explain():
    data = request.json
    concept = data.get('concept')
    interest = data.get('interest')
    language = data.get('language', 'English')
    level = data.get('level', 'a school student')

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

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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

    return jsonify({"explanation": response.choices[0].message.content})


if __name__ == '__main__':
    app.run(debug=True)
