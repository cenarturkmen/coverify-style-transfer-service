from flask import Flask, jsonify, request
from model.Classifier.ImagenetClassifier import ImagenetClassifier
from model.StyleTransfer.StyleTransfer import StyleTransfer

app = Flask(__name__)
model = ImagenetClassifier()
styleTransferModel = StyleTransfer()


@app.route("/health", methods=["GET"])
def health():
    return "<a>Service is up and running.</a>"


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img_bytes = file.read()
    class_id, class_name = model.get_prediction(image_bytes=img_bytes)
    return jsonify({"class_id": class_id, "class_name": class_name})


@app.route("/transfer", methods=["POST"])
def transfer():
    content = request.files["content"]
    style = request.files["style"]
    content_bytes = content.read()
    style_bytes = style.read()

    output = styleTransferModel.transfer_style(content_bytes, style_bytes)
    return jsonify({"img": output})


if __name__ == "__main__":
    app.run()
