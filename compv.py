# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests

# # load image from the IAM database
# # url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# # image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# image = Image.open("images/if.jpg").convert("RGB")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(generated_text)

from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()
custom_objects = detector.CustomObjects(person=True, car=False)
detections = detector.detectCustomObjectsFromImage(input_image=os.path.join(execution_path , "image.png"), output_image_path=os.path.join(execution_path , "image_new.png"), custom_objects=custom_objects, minimum_percentage_probability=65)


for eachObject in detections:
   print(eachObject["name"] + " : " + eachObject["percentage_probability"] )
   print("--------------------------------")
