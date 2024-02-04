import cv2
from imutils import contours
import imutils
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import ToTensor
import classifier

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

    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])  
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image)
    
    input_tensor = input_tensor.unsqueeze(0)

    model = classifier.MyResNet()
    model.load_state_dict(torch.load('model/model.pt'))
    
    prediction = model(input_tensor)
    prediction = torch.nn.functional.softmax(prediction)
    print(prediction)
    output = torch.argmax(prediction)
    code = get_code(Tensor.tolist(output))
    print(code)
    return code


def get_code(idx):
    labels = ["else","if","while","print","for","x","y","+","=","and","or","not","divisible","1","2","5"]

    label_dict = {}
    for (i, label) in enumerate(labels):
        label_dict[i] = label
    return label_dict[idx]

def get_code_updated(idx):
    labels = ["else","if","while","print","repeat","add","sub","mult","div","=","and","or","not","divisible","1","2","3","4","5","6","7","8","9","0","do","end","gt","lt"]

    label_dict = {}
    for (i, label) in enumerate(labels):
        label_dict[i] = label

    return label_dict[idx]
    

if __name__ == '__main__':
    image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/prog2.jpg')
    box_paths = save_bounding_boxes(image)
    code_tokens = []
    for l in box_paths:
        line = []
        for box_path in l:
            token = get_prediction(box_path)
            line.append(token)
        code_tokens.append(line)

    print(code_tokens)