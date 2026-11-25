from flask import Flask, request, jsonify
from rag_tool import search_university_knowledge

app = Flask(__name__)


@app.get("/")
def home():
    return jsonify({"status": "RAG server is running", "usage": "/search?q=your question"})


@app.get("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing query parameter q"}), 400
    result = search_university_knowledge.invoke({"query": q})
    return jsonify({"result": result})


if __name__ == "__main__":
    # 0.0.0.0 so the n8n Docker container can reach it via host.docker.internal
    app.run(host="0.0.0.0", port=8000)
