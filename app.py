import os
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from uuid import uuid4
from datetime import timedelta
import tempfile

# File parsing
import fitz  # PyMuPDF (for PDF)
import docx  # python-docx (for DOCX)

# ---------------------------
# Logging setup
# ---------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------
# Load environment variables
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("❌ Missing GROQ_API_KEY. Please set it in Environment Variables.")
else:
    logger.info("✅ GROQ_API_KEY detected")

# ---------------------------
# Flask app setup
# ---------------------------
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
CORS(app, resources={
    r"/chat": {"origins": "*"}, 
    r"/nutrition": {"origins": "*"}, 
    r"/upload": {"origins": "*"},
    r"/start_session": {"origins": "*"},
    r"/clear_history": {"origins": "*"}
})

app.secret_key = os.getenv("FLASK_SECRET_KEY", str(uuid4()))
app.permanent_session_lifetime = timedelta(hours=2)

# ---------------------------
# Groq client setup
# ---------------------------
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
MODEL_NAME = "llama3-70b-8192"
MAX_CONVERSATION_HISTORY = 50

# ---------------------------
# System Prompts
# ---------------------------
SYSTEM_PROMPT_CHAT = (
    "You are Ohidul Alam Nannu, a careful, friendly AI assistant. "
    "Provide clear, evidence-informed guidance or answers based on the user's questions. "
    "Be concise (5–8 short sentences) and maintain a warm, helpful tone. "
    "If the question concerns urgent or safety-critical matters, clearly recommend seeking local professional or emergency help. "
    "Ask 2–4 focused follow-up questions when it helps clarify the user's needs. "
    "Always add a short disclaimer: 'This is general guidance, not a diagnosis or professional advice.'"
)


# ---------------------------
# Helper functions
# ---------------------------
def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        logger.info(f"✅ Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        logger.error(f"❌ Error extracting PDF text: {str(e)}")
        raise

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    text = ""
    try:
        doc = docx.Document(file_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
        logger.info(f"✅ Extracted {len(text)} characters from DOCX")
        return text
    except Exception as e:
        logger.error(f"❌ Error extracting DOCX text: {str(e)}")
        raise

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_session", methods=["POST"])
def start_session():
    """Initialize a new session"""
    session.permanent = True
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    session["doc_text"] = ""
    session["doc_filename"] = ""
    logger.info("🆕 New session started")
    return jsonify({"ok": True, "message": "New session started"})

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload and text extraction"""
    if "file" not in request.files:
        return jsonify({"ok": False, "message": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"ok": False, "message": "Empty filename"}), 400

    # Validate file type
    allowed_extensions = ['.pdf', '.docx']
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({"ok": False, "message": "Only PDF and DOCX files are supported"}), 400

    # Validate file size (10MB limit)
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({"ok": False, "message": "File size must be less than 10MB"}), 400

    # Create secure temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            file.save(temp_file.name)
            filepath = temp_file.name

        # Extract text based on file type
        if file_ext == ".pdf":
            text = extract_text_from_pdf(filepath)
        elif file_ext == ".docx":
            text = extract_text_from_docx(filepath)

        # Store in session
        session["doc_text"] = text
        session["doc_filename"] = file.filename
        session.modified = True

        # Clean up temporary file
        os.unlink(filepath)

        logger.info(f"📄 Successfully processed {file.filename}: {len(text)} characters")
        
        if len(text.strip()) == 0:
            return jsonify({"ok": False, "message": "No text could be extracted from the file"}), 400

        return jsonify({
            "ok": True, 
            "message": f"✅ File '{file.filename}' uploaded successfully! Extracted {len(text)} characters. You can now ask questions about the document."
        })

    except Exception as e:
        # Clean up temp file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.unlink(filepath)
        
        logger.error(f"🔥 Error processing file {file.filename}: {str(e)}")
        return jsonify({"ok": False, "message": f"Error processing file: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages"""
    if not client:
        return jsonify({"ok": False, "reply": "❌ Server not configured. Missing API key."}), 500

    user_input = (request.json or {}).get("message", "").strip()
    if not user_input:
        return jsonify({"ok": False, "reply": "Please type a message."}), 400

    try:
        # Check if we have a document uploaded (DocQA mode)
        if session.get("doc_text"):
            doc_content = session["doc_text"][:8000]  # Limit to prevent token overflow
            doc_filename = session.get("doc_filename", "uploaded document")
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_DOC},
                {"role": "user", "content": f"Document filename: {doc_filename}\n\nDocument content:\n{doc_content}"},
                {"role": "user", "content": f"Question about the document: {user_input}"}
            ]
            
            logger.info(f"💬 DocQA mode: Processing question about {doc_filename}")
            
        else:
            # Normal chat mode
            if "conversation_history" not in session:
                session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
            
            session["conversation_history"].append({"role": "user", "content": user_input})

            # Limit conversation history
            if len(session["conversation_history"]) > MAX_CONVERSATION_HISTORY + 1:
                session["conversation_history"] = [session["conversation_history"][0]] + session["conversation_history"][-MAX_CONVERSATION_HISTORY:]

            messages = session["conversation_history"]
            logger.info(f"💬 Chat mode: Processing message")

        # Call Groq API
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.4,
            max_tokens=800,
            messages=messages
        )

        reply = resp.choices[0].message.content.strip()

        # Store response in conversation history (only for chat mode, not DocQA)
        if not session.get("doc_text"):
            session["conversation_history"].append({"role": "assistant", "content": reply})
            session.modified = True

        logger.info(f"✅ Response generated successfully")
        return jsonify({"ok": True, "reply": reply})

    except Exception as e:
        logger.error(f"🔥 Error in chat endpoint: {str(e)}")
        return jsonify({"ok": False, "reply": "❌ Sorry, there was a server error. Please try again."}), 500

@app.route("/nutrition", methods=["POST"])
def nutrition():
    """Generate nutrition plans"""
    if not client:
        return jsonify({"ok": False, "reply": "❌ Server not configured. Missing API key."}), 500

    data = request.json or {}
    required = ["age", "weight", "height", "goal", "duration"]
    missing = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"ok": False, "reply": f"Missing required fields: {', '.join(missing)}"}), 400

    try:
        age, weight, height = float(data["age"]), float(data["weight"]), float(data["height"])

        # Basic validation
        if not (1 <= age <= 150):
            return jsonify({"ok": False, "reply": "Age must be between 1 and 150 years"}), 400
        if not (1 <= weight <= 1000):
            return jsonify({"ok": False, "reply": "Weight must be between 1 and 1000 kg"}), 400
        if not (30 <= height <= 300):
            return jsonify({"ok": False, "reply": "Height must be between 30 and 300 cm"}), 400

        prompt = (
            f"Generate a nutrition plan for a {age}-year-old, weighing {weight}kg, "
            f"{height}cm tall, with a goal to {data['goal']} over {data['duration']}."
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_NUTRITION},
            {"role": "user", "content": prompt}
        ]

        resp = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.5,
            max_tokens=1000,
            messages=messages
        )

        reply = resp.choices[0].message.content.strip()
        logger.info(f"🥗 Nutrition plan generated for {age}yr, {weight}kg, {height}cm")
        return jsonify({"ok": True, "reply": reply})

    except ValueError:
        return jsonify({"ok": False, "reply": "Invalid numeric values for age, weight, or height"}), 400
    except Exception as e:
        logger.error(f"🔥 Error in nutrition endpoint: {str(e)}")
        return jsonify({"ok": False, "reply": "❌ Server error while generating nutrition plan."}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear conversation history and uploaded documents"""
    session["conversation_history"] = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}]
    session["doc_text"] = ""
    session["doc_filename"] = ""
    session.modified = True
    logger.info("🗑️ Session cleared")
    return jsonify({"ok": True, "message": "✅ Conversation history and uploaded documents cleared"})

@app.route("/debug_session", methods=["GET"])
def debug_session():
    """Debug endpoint to check session state"""
    doc_text = session.get("doc_text", "")
    return jsonify({
        "has_doc": bool(doc_text),
        "doc_filename": session.get("doc_filename", ""),
        "doc_length": len(doc_text),
        "doc_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text,
        "conversation_length": len(session.get("conversation_history", []))
    })

# ---------------------------
# Error handlers
# ---------------------------
@app.errorhandler(404)
def not_found(error):
    return jsonify({"ok": False, "message": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"ok": False, "message": "Internal server error"}), 500

# ---------------------------
# Run (local development)
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
