import io
import json
from numpy import array
from torchvision import models
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class ImagenetClassifier:
    def __init__(self) -> None:
        self.imagenet_class_index = json.load(open("model/imagenet_class_index.json"))
        self.model = torchvision.models.densenet121(
            weights=torchvision.models.DenseNet121_Weights.DEFAULT
        )
        self.model.eval()

    def transform_image(self, image_bytes) -> array:
        my_transforms = transforms.Compose(
            [
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = Image.open(io.BytesIO(image_bytes))
        return my_transforms(image).unsqueeze(0)

    def get_prediction(self, image_bytes) -> str:
        tensor = self.transform_image(image_bytes=image_bytes)
        outputs = self.model.forward(tensor)
        _, y_hat = outputs.max(1)
        predicted_idx = str(y_hat.item())
        return self.imagenet_class_index[predicted_idx]
