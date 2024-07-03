import io
from flask import Flask, render_template, request, jsonify
import soundfile as sf
from functions import image2text, text2audio, textify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
  return render_template('index.html')

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        try:
            image_file = request.files["image"]

            if image_file.filename != '':
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)

                image_file.save(image_path)

                extracted_text = image2text(image_path)

                if extracted_text:

                    proper_text = textify(extracted_text)

                    audio_data = text2audio(proper_text)

                    if audio_data:
                        audio_bytes = io.BytesIO()
                        sf.write(audio_bytes, audio_data["audio"], samplerate=audio_data["sampling_rate"])
                        audio_bytes_data = audio_bytes.getvalue()
                        audio_bytes.close() 

                        return jsonify({"success": True, "message": "Text converted to speech", "audio_bytes": audio_bytes_data})
                    else:
                        return jsonify({"success": False, "message": "Error generating audio"})
                else:
                    return jsonify({"success": False, "message": "Error extracting text from image"})

            else:
                return jsonify({"success": False, "message": "No file selected"})

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({"success": False, "message": "An error occurred. Please try again."})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
