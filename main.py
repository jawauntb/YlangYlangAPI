import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from ylangchain import define_tools, create_search_agent, run_agent_executor
from replit import db
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_API_KEY")

# db['ylang_queries']
# db['documents']

app = Flask(__name__)
CORS(app)


def encode_file_contents(file):
  file_contents = file.read()
  return base64.b64encode(file_contents).decode('utf-8')


def decode_file_contents(file_contents):
  return base64.b64decode(file_contents.encode('utf-8'))


@app.route('/db', methods=['GET'])
def get_db():
  return jsonify(db['ylang_queries'])


@app.route('/docs', methods=['GET', 'POST'])
def handle_docs():
  documents = db["documents"]
  if request.method == 'GET':
    documents_copy = [dict(document) for document in documents]
    for document in documents_copy:
      if 'file_contents' in document:
        decoded_doc = decode_file_contents(document['file_contents'])
        document['file_contents'] = decoded_doc.decode('utf-8')
    return jsonify(documents_copy)
  elif request.method == 'POST':
    file = request.files['file']
    filename = file.filename
    file_contents = encode_file_contents(file.read())
    document = {
      'id': len(documents),
      'name': filename,
      'file_contents': file_contents
    }
    documents.append(document)
    db["documents"] = documents
    return jsonify(document), 201


@app.route('/docs/<int:doc_id>', methods=['DELETE'])
def handle_doc(doc_id):
  documents = db["documents"]
  document = next((d for d in documents if d['id'] == doc_id), None)
  if not document:
    return 'Document not found', 404
  documents.remove(document)
  db["documents"] = documents
  return '', 204


@app.route('/docs/int:doc_id/edits', methods=['PATCH'])
def handle_doc_edits(doc_id):
  print('handle_doc_edits')
  documents = db["documents"]
  if not documents:
    print('no documents in thing')
    return 'Documents not found', 404
  document = next((d for d in documents if d['id'] == doc_id), None)
  if not document:
    print('no document at doc id', doc_id)
    return 'Document not found', 404
  data = request.get_json()
  document.update(data)
  db["documents"] = documents
  response = jsonify(document)
  response.headers.add('Access-Control-Allow-Origin', '*')
  response.headers.add('Access-Control-Allow-Methods', 'PATCH')
  response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
  response.headers.add('Access-Control-Max-Age', 60 * 60 * 24 * 20)
  return response


@app.route('/ylang', methods=['POST'])
def receive_input():
  query = request.get_json().get('query')
  tools = define_tools()
  search_agent = create_search_agent(tools)
  result = run_agent_executor(search_agent, tools, query=query)

  db['ylang_queries'].append({'query': query, 'result': result})
  return jsonify(result)


def run_query_document(query, file_contents, openai_key):
  document = file_contents
  text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
  texts = text_splitter.split_text(document)
  embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
  docsearch = FAISS.from_texts(texts, embeddings)
  docs = docsearch.similarity_search(query)
  return docs


@app.route('/docs/int:doc_id/query', methods=['POST'])
def query_document(doc_id):
  documents = db["documents"]
  document = documents[doc_id]
  next((d for d in documents if d['id'] == doc_id), None)
  if not document:
    return 'Document not found', 404
  query = request.get_json().get('query')
  file_contents = decode_file_contents(document['file_contents']).decode()
  openai_key = os.getenv("OPENAI_API_KEY")
  result = run_query_document(query, file_contents, openai_key)
  return jsonify(result)


@app.route('/')
def index():
  return 'Hello from Flask!'


if __name__ == "__main__":
  app.run(host='0.0.0.0', port=81)
