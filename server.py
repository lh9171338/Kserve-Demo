# -*- encoding: utf-8 -*-
"""
@File    :   server.py
@Time    :   2025/03/20 15:13:34
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
import base64
import yaml
import io
from PIL import Image
import torch
import torchvision
import kserve
import logging


# set base logging config
fmt = "[%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=fmt, level=logging.INFO)


class ResNetModel(kserve.Model):
    """
    ResNet Model
    """

    def __init__(self, name, class_name_file, test_image_path=None):
        super().__init__(name)
        self.name = name
        self.ready = False
        self.test_image_path = test_image_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_name_file = class_name_file

        # load model
        self.load()

    def load(self):
        """load"""
        # load class_name_file
        with open(self.class_name_file, "r") as f:
            self.class_names = yaml.safe_load(f)

        # build model
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"number of params: {n_parameters / (1024 ** 2):.2f}M")

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        self.ready = True
        return self.ready

    def preprocess(self, inputs, headers=None):
        """
        preprocess

        Args:
            inputs (list[dict]): input request
            headers (dict): request headers1

        Returns:
            images (Tensor): image with shape [B, C, H, W]
        """
        if not isinstance(inputs, list):
            inputs = [inputs]

        images = []
        for input in inputs:
            image = Image.open(io.BytesIO(base64.b64decode(input["image"])))
            logging.info(f"image shape: {image.size}")

            image = self.transform(image)
            images.append(image)

        images = torch.stack(images)
        return images

    def predict(self, images, headers=None):
        """
        predict

        Args:
            images (Tensor): preprocessed image tensor with shape [B, C, H, W]
            headers (dict): request headers

        Returns:
            outputs (Tensor): classification logits with shape [B, C]
        """
        with torch.no_grad():
            outputs = self.model(images)
            outputs = outputs.softmax(dim=1)

        return outputs

    def postprocess(self, outputs, headers=None):
        """
        postprocess

        Args:
            outputs (Tensor): classification logits with shape [B, C]
            headers (dict): request headers

        Returns:
            results (list[dict]): response
        """
        scores, indices = outputs.max(dim=1)
        scores = scores.cpu().numpy().tolist()
        indices = indices.cpu().numpy().tolist()
        results = []
        for score, index in zip(scores, indices):
            class_name = self.class_names[index]
            result = dict(
                score=score,
                label=index,
                class_name=class_name,
            )
            results.append(result)

        return results

    def test(self):
        """test"""
        assert self.test_image_path is not None, "`test_image_path` is None"
        image_files = os.listdir(self.test_image_path)
        inputs = []
        for image_file in image_files:
            image_file = os.path.join(self.test_image_path, image_file)
            image = Image.open(image_file)
            byte_arr = io.BytesIO()
            image.save(byte_arr, format="PNG")
            image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
            input = dict(image=image)
            inputs.append(input)

        images = self.preprocess(inputs)
        outputs = self.predict(images)
        results = self.postprocess(outputs)
        for image_file, result in zip(image_files, results):
            print(f"{image_file}: {result}")


if __name__ == "__main__":
    model = ResNetModel(
        name="ResNetModel",
        class_name_file="imagenet_classes.yaml",
        test_image_path="images",
    )
    kserve.ModelServer().start([model])
