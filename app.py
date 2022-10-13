import base64
from io import BytesIO
from black import out
from flask import Flask, jsonify, request
from model.ImagenetClassifier import ImagenetClassifier
from model.StyleTransfer import output

app = Flask(__name__)
model = ImagenetClassifier()


@app.route("/health", methods=["GET"])
def health():
    return "<a>Service is up and running.</a>"


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        img_bytes = file.read()
        class_id, class_name = model.get_prediction(image_bytes=img_bytes)
        return jsonify({"class_id": class_id, "class_name": class_name})


@app.route("/transfer", methods=["GET"])
def transfer():
    # rint(outputTransformed)
    print(output)
    return "Zort"


if __name__ == "__main__":
    app.run()
