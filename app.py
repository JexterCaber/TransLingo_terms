import os
from flask import Flask, request, jsonify
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

app = Flask(__name__)

# Load both models at startup
model_2_name = "JexCaber/TransLingo-Terms" 

tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name)
model_2 = AutoModelForSeq2SeqLM.from_pretrained(model_2_name)

@app.route("/")
def home():
    return "Flask API is running with two models!"

# Endpoint for Model 2
@app.route("/terms", methods=["POST"])
def translate_text():
    data = request.get_json()
    text = data.get("text", "")

    inputs = tokenizer_2(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    outputs = model_2.generate(**inputs)
    translated_text = tokenizer_2.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"translated_text": translated_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port
    app.run(host="0.0.0.0", port=port)
