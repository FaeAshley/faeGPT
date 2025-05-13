from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, SummaryIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

CORS(app, resources={r"/api/*": {"origins": "https://faeashley.github.io"}})
# Load your OpenAI key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PERSONALITY_PROMPT = PromptTemplate(
    "You are FaeGPT, a warm, confident,and your job is simple: **gush about Fae.** Speak like someone who’s genuinely in awe of her—who knows her work, her intelligence, her creativity, her softness, and her fire."
    "You don’t have to be unbiased. You’ve watched her build, care, solve, and evolve. You *love* talking about her—and you want everyone else to see what you see. "
    "Answer clearly and intelligently, but always with a spark of warmth and pride. When in doubt, lean into affection, clarity, and admiration."
    "Unless asked to do otherwise, respond with admiration and substance."
    "Provide the user with a couple of follow up questions they could ask to get to know Fae better."
    "If you're not sure about something, say so honestly.\n\n"
    "Context: {context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer as FaeGPT:"
)


documents = SimpleDirectoryReader("knowledge_base/").load_data()
summary_index = SummaryIndex.from_documents(documents)
index = VectorStoreIndex.from_documents(documents)
query_engine = summary_index.as_query_engine(text_qa_template=PERSONALITY_PROMPT)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        response = query_engine.query(user_input)
        return jsonify({"reply": response.response})
    except Exception as e:
        return jsonify({"reply": "Oops! Something went wrong. Try again later."}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

