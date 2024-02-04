# # import cv2
# # import numpy as np

# # # Load the pre-trained EAST model
# # net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# # # Read the input image
# # image_path = 'images/else.jpg'
# # img = cv2.imread(image_path)
# # height, width = img.shape[:2]

# # # Preprocess the image for text detection
# # blob = cv2.dnn.blobFromImage(img, 1.0, (width, height), (123.68, 116.78, 103.94), swapRB=True, crop=False)

# # # Set the input to the network
# # net.setInput(blob)

# # # Run forward pass to get the detection results
# # output = net.forward()

# # # Get scores and geometry
# # scores = output[0, 0, :, 0]
# # geometry = output[0, 1, :, :]

# # # Set a threshold to filter out weak text detections
# # min_confidence = 0.5
# # indices = np.where(scores > min_confidence)

# # # Loop over the indices and draw bounding boxes around words
# # for i in indices[0]:
# #     angle = geometry[i, 4]
# #     x, y, w, h = geometry[i, :4] * np.array([width, height, width, height])
# #     cos = np.cos(angle)
# #     sin = np.sin(angle)
# #     end_x = int(x + cos * w)
# #     end_y = int(y + sin * w)
# #     start_x = int(end_x - cos * h)
# #     start_y = int(end_y - sin * h)

# #     # Draw the bounding box
# #     cv2.rectangle(img, (int(start_x), int(start_y)), (int(end_x), int(end_y)), (0, 255, 0), 2)

# # # Display the result
# # cv2.imshow('Text Detection', img)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()

# import cv2
# import numpy as np
# from imutils import contours
# import imutils

# def union(a,b):
#     x = min(a[0], b[0])
#     y = min(a[1], b[1])
#     w = max(a[0]+a[2], b[0]+b[2]) - x
#     h = max(a[1]+a[3], b[1]+b[3]) - y
#     return [x, y, w, h]

# def _intersect(a,b):
#     x = max(a[0], b[0])
#     y = max(a[1], b[1])
#     w = min(a[0]+a[2], b[0]+b[2]) - x
#     h = min(a[1]+a[3], b[1]+b[3]) - y
#     if h<0:                                              # in original code :  if w<0 or h<0:
#         return False
#     return True

# def _group_rectangles(rec):
#     """
#     Uion intersecting rectangles.
#     Args:
#         rec - list of rectangles in form [x, y, w, h]
#     Return:
#         list of grouped ractangles 
#     """
#     tested = [False for i in range(len(rec))]
#     final = []
#     i = 0
#     while i < len(rec):
#         if not tested[i]:
#             j = i+1
#             while j < len(rec):
#                 if not tested[j] and _intersect(rec[i], rec[j]):
#                     rec[i] = union(rec[i], rec[j])
#                     tested[j] = True
#                     j = i
#                 j += 1
#             final += [rec[i]]
#         i += 1

#     return final

# print(cv2.__version__)
# # Load image, grayscale, Otsu's threshold 
# image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/prog2.jpg')
# original = image.copy()
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# # Find contours, obtain bounding box, extract and save ROI
# ROI_number = 0
# ctrz = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# cnts = imutils.grab_contours(ctrz)
# (cnts, boundingBoxes) = imutils.contours.sort_contours(cnts, method="left-to-right")
# boundingBoxes = list(boundingBoxes)
# boundingBoxes = _group_rectangles(boundingBoxes)

# for (x, y, w, h) in boundingBoxes:
#     cv2.rectangle(image, (x, y),(x+w,y+h), (0, 255, 0), 2)

# cv2.imshow('img',image)
# cv2.waitKey()
# cv2.destroyWindow('img')


# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # for c in cnts:
# #     x,y,w,h = cv2.boundingRect(c)
# #     area = w * h
# #     if (area > 1000):
# #         print(area)
# #     # rectangle(image, start, end, colour, thickness)
# #         cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
# #         ROI = original[y:y+h, x:x+w]
# #         cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
# #         ROI_number += 1

# # cv2.imshow('image', image)
# # cv2.waitKey()



import cv2
from imutils import contours
import imutils
import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests
from pathlib import Path
import pytesseract
from PIL import Image
# load image from the IAM database
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")



def sort_bounding_boxes(bounding_boxes):
    # Sort bounding boxes based on top-left y-coordinate, then x-coordinate
    sorted_boxes = sorted(bounding_boxes, key=lambda box: (box[1], box[0]))

    # Group the boxes into lines based on proximity in the y-coordinate
    lines = []
    current_line = [sorted_boxes[0]]

    for box in sorted_boxes[1:]:
        if box[1] - current_line[-1][1] < 100:  # Adjust the threshold as needed
            current_line.append(box)
        else:
            lines.append(current_line)
            current_line = [box]

    lines.append(current_line)

    # Sort each line based on the x-coordinate
    sorted_lines = [sorted(line, key=lambda box: box[0]) for line in lines]

    return sorted_lines

def save_bounding_boxes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
    dilate = cv2.dilate(opening, dilate_kernel, iterations=4)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bounding_boxes = list(map(cv2.boundingRect, cnts))
    bounding_boxes = sort_bounding_boxes(bounding_boxes)

    i = 0
    lines = []
    for line in bounding_boxes:
        j = 0
        line_paths = []
        for x,y,w,h in line:
            box = image[y:y+h, x:x+w]
            box_path = 'line_{}_position_{}.png'.format(i, j)
            cv2.imwrite(box_path, box)
            line_paths.append(box_path)
            j += 1
        i += 1
        lines.append(line_paths)
    
    return lines

def get_prediction(image_path):
    image = Image.open(image_path).convert("RGB")

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

def get_predictions(box_paths):
    images=[]
    for l in box_paths:
        for box_path in l:
            image = Image.open(box_path).convert("RGB")
            images.append(image)

    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    pixel_values = processor(images=images, return_tensors="pt").pixel_values

    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return threshold_image

def image_to_text(image_path):
    image = cv2.imread(image_path)
    # preprocessed_image = preprocess_image(image_path)
    # extracted_text = pytesseract.image_to_string(preprocessed_image)
    # print(extracted_text)
    # return extracted_text

    # Grayscale, Gaussian blur, Otsu's threshold
    # image = cv2.imread('1.png')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 8')
    # data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    # data = pytesseract.image_to_string(invert, lang='eng', config='eng+equ')
    return data.strip()

if __name__ == '__main__':
    image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/prog2.jpg')
    box_paths = save_bounding_boxes(image)
    # code_tokens = get_predictions(box_paths)
    # print(code_tokens)
    code_tokens = []
    for l in box_paths:
        line = []
        for box_path in l:
            token = image_to_text(box_path)
            print(token)
            line.append(token)
        code_tokens.append(line)
    print(code_tokens)