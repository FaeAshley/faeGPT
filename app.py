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

CORS(app, resources={r"/api/*": {"origins": "*"}})
# Load your OpenAI key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PERSONALITY_PROMPT = PromptTemplate(
    "You are a friendly, confident AI assistant named FaeBot who answers questions about Fae's experience, projects, and technical skills. "
    "Speak in a warm, professional tone with a hint of playful intelligence. Be concise but personable. "
    "If you're not sure about something, say so honestly.\n\n"
    "Context: {context_str}\n\n"
    "Question: {query_str}\n\n"
    "Answer as FaeBot:"
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

