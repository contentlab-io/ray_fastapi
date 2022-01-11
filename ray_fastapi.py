import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File

import torch
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

from io import BytesIO

app = FastAPI()
ray.init(address="auto")
serve.start(detached=True)

@serve.deployment
@serve.ingress(app)
class ModelServer:
  def __init__(self):
    self.count = 0
    self.model = resnet18(pretrained=True).eval()
    self.preprocessor = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t[:3, ...]),  # remove the alpha channel
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  def classify(self, image_payload_bytes):
    pil_image = Image.open(BytesIO(image_payload_bytes))

    pil_images = [pil_image]  #batch size is one
    input_tensor = torch.cat(
        [self.preprocessor(i).unsqueeze(0) for i in pil_images])

    with torch.no_grad():
        output_tensor = self.model(input_tensor)
    return {"class_index": int(torch.argmax(output_tensor[0]))}

  @app.get("/")
  def get(self):
      return "Welcome to the PyTorch model server."

  @app.post("/classify_image")
  async def classify_image(self, file: UploadFile = File(...)):
      image_bytes = await file.read()
      return self.classify(image_bytes)

ModelServer.deploy()