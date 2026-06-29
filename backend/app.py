from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from flask_cors import CORS

from routes.generate import generate_bp
from routes.substitute import substitute_bp
from routes.generate_text import generate_text_bp
from routes.generate_video import generate_video_bp

app = Flask(__name__)
CORS(app)

app.register_blueprint(generate_bp)
app.register_blueprint(substitute_bp)
app.register_blueprint(generate_text_bp)
app.register_blueprint(generate_video_bp)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

    