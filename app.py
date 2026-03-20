from flask import Flask, render_template, request
import os
from detect import detect_traffic

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/uploads/result"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():

    result = None
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filename = file.filename

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            result = detect_traffic(filepath)

            if result:
                image_path = result["output_image"]

    return render_template(
        "index.html",
        result=result,
        image_path=image_path
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
