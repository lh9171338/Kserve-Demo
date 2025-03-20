# -*- encoding: utf-8 -*-
"""
@File    :   service.py
@Time    :   2025/03/20 15:14:26
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
import requests
import base64
import argparse
import io
from PIL import Image


def call_service(ip, image_path):
    """
    call service

    Args:
        ip (str): ip
        image_path (str): image path

    Returns:
        None
    """
    # load images
    image_files = os.listdir("images")
    inputs = []
    for image_file in image_files:
        image_file = os.path.join(image_path, image_file)
        image = Image.open(image_file)
        byte_arr = io.BytesIO()
        image.save(byte_arr, format="PNG")
        image = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
        input = dict(image=image)
        inputs.append(input)

    # call service
    response = requests.post(f"http://{ip}:8080/v1/models/ResNetModel:predict", json=inputs)
    assert response.status_code == 200, f"Failed to call service, get status code {response.status_code}"
    results = response.json()

    for image_file, result in zip(image_files, results):
        print(f"{image_file}: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, help="server ip", default="127.0.0.1")
    parser.add_argument("--image_path", type=str, help="image path", default="images")
    args = parser.parse_args()

    call_service(ip=args.ip, image_path=args.image_path)
