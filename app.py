from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os
import re
from ecommbot.retrieval_generation import generation
from ecommbot.ingest import ingestdata
from celery import Celery

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    return celery

load_dotenv()

celery = make_celery(app)

vstore, inserted_ids = ingestdata()
conversational_rag_chain = generation(vstore)

def format_bold_and_list_text(text):
    # Convert **text** to <strong>text</strong> for HTML bold formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    # Convert lines starting with '* ' into HTML list items
    text = re.sub(r'\* (.*?)\n', r'<li>\1</li>', text)
    # Wrap the entire list with <ul> tags if any <li> tags are present
    if "<li>" in text:
        text = "<ul>" + text + "</ul>"
    # Replace remaining newlines with <br> for other line breaks
    return text.replace('\n', '<br>')

@app.route("/")
def index():
    return render_template('chat.html')

# Celery task for generating response
@celery.task
def generate_response_task(msg):
    input = msg

    result = conversational_rag_chain.invoke({"input": input},
                config={
                        "configurable": {"session_id": "abc123"}
                        },  # constructs a key "abc123" in `store`.
                )["answer"]
    
    # Format the result text to handle bold, lists, and line breaks
    result_html = format_bold_and_list_text(result)

    # Embed the formatted text in a styled HTML structure
    formatted_result = (
        "<div style='font-family: Arial, sans-serif; color: #333; line-height: 1.6;'>"
        "<p><strong>Response:</strong></p>"
        "<div style='padding: 10px; border: 1px solid #ddd; background-color: #f9f9f9; border-radius: 8px;'>"
        f"{result_html}"
        "</div>"
        "</div>"
    )
    return formatted_result

# Start the background task and return task_id
@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("msg")
    task = generate_response_task.delay(msg)

    return jsonify({"task_id": task.id}), 202

# Polling endpoint to retrieve task result
@app.route("/result/<task_id>")
def get_result(task_id):
    task = generate_response_task.AsyncResult(task_id)
    if task.state == 'SUCCESS':
        return jsonify({"Response": task.result}), 200
    elif task.state == 'PENDING':
        return jsonify({"status": "processing"}), 202
    else:
        return jsonify({"status": "error"}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0")