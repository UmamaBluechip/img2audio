from flask import Flask, render_template, request, jsonify
import soundfile as sf
from functions import image2text, text2audio, textify

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        try:
            image_file = request.files["image"]

            extracted_text = image2text(image_file.filename)

            if extracted_text:

                proper_text = textify(extracted_text)

                audio = text2audio(proper_text)

                audio_bytes = sf.write("audio.wav", audio["audio"], samplerate=audio["sampling_rate"])

                if audio_bytes:
                    return jsonify({"success": True, "message": "Text converted to audio", "audio_bytes": audio_bytes})
                else:
                    return jsonify({"success": False, "message": "Error generating audio"})
            else:
                return jsonify({"success": False, "message": "Error extracting text from image"})

        except Exception as e:
            print(f"Error during processing: {e}")
            return jsonify({"success": False, "message": "An error occurred. Please try again."})

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
