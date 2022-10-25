from model.StyleTransfer.ContentLoss import ContentLoss
from model.StyleTransfer.StyleLoss import StyleLoss
from model.StyleTransfer.Normalization import Normalization

import base64
from io import BytesIO
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models


class StyleTransfer:
    def __init__(self) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imsize = 128  # use small size if no gpu
        self.cnn = (
            models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to("cpu").eval()
        )

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(
            self.device
        )
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)

        # desired depth layers to compute style/content losses :
        self.content_layers_default = ["conv_4"]
        self.style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

    # transform it into a torch tensor
    def image_loader(self, image_name):

        loader = transforms.Compose(
            [
                transforms.Resize(self.imsize),
                transforms.ToTensor(),
            ]  # scale imported image
        )
        image = Image.open(image_name)
        image = image.resize((128, 128))
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def get_style_model_and_losses(
        self,
        style_img,
        content_img,
    ):
        # normalization module
        normalization = Normalization(
            self.cnn_normalization_mean, self.cnn_normalization_std
        ).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in self.cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_{}".format(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu_{}".format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_{}".format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = "bn_{}".format(i)
            else:
                raise RuntimeError(
                    "Unrecognized layer: {}".format(layer.__class__.__name__)
                )

            model.add_module(name, layer)

            if name in self.content_layers_default:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers_default:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[: (i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img])
        return optimizer

    def run_style_transfer(
        self,
        content_img,
        style_img,
        input_img,
        num_steps=10,
        style_weight=1000000,
        content_weight=1,
    ):
        """Run the style transfer."""
        print("Building the style transfer model..")

        model, style_losses, content_losses = self.get_style_model_and_losses(
            style_img, content_img
        )

        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        input_img.requires_grad_(True)
        model.requires_grad_(False)

        optimizer = self.get_input_optimizer(input_img)

        print("Optimizing..")
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                with torch.no_grad():
                    input_img.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print(
                        "Style Loss : {:4f} Content Loss: {:4f}".format(
                            style_score.item(), content_score.item()
                        )
                    )
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        with torch.no_grad():
            input_img.clamp_(0, 1)

        return input_img

    def transform_image(self, image_bytes):
        loader = transforms.Compose(
            [
                transforms.Resize(self.imsize),
                transforms.ToTensor(),
            ]  # scale imported image
        )  # transform it into a torch tensor

        image = Image.open(BytesIO(image_bytes))
        # fake batch dimension required to fit network's input dimensions
        image = loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def transfer_style(self, content_img, style_img):
        content_img = self.transform_image(content_img)
        style_img = self.transform_image(style_img)

        input_img = content_img.clone()

        output = self.run_style_transfer(
            content_img,
            style_img,
            input_img,
        )

        toPil = transforms.ToPILImage()(output.squeeze(0))
        buffered = BytesIO()
        toPil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        img_base64 = bytes("data:image/jpeg;base64,", encoding="utf-8") + img_str

        return img_base64.decode("utf-8")
