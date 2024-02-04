import cv2
from imutils import contours
import imutils
import numpy as np

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

# def get_prediction(input_path: str) -> str:
#     return ""

image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/repeat_20.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray, 3)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,2))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
dilate = cv2.dilate(opening, dilate_kernel, iterations=4)
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ctrz = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(ctrz)
# (cnts, boundingBoxes) = imutils.contours.sort_contours(cnts, method="left-to-right")
# for c in cnts:
#     print(c)
# print("Bounding boxes:")
# for c in boundingBoxes:
#     print(c)

# i = 0
# for x,y,w,h in boundingBoxes:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#     cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#     print(x, y)
#     i += 1
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
bounding_boxes = list(map(cv2.boundingRect, cnts))
print(len(bounding_boxes))
bounding_boxes = list(filter(lambda b : b[2] * b[3] > 5000, bounding_boxes))
print(">>>")
for b in bounding_boxes:
    print(b[3] * b[2])
print(">>>")

print(len(bounding_boxes))
bounding_boxes = sort_bounding_boxes(bounding_boxes)

print(">>>")
for b in bounding_boxes:
    for x,y,w,h in b:
        print(w*h)
    # print(w*h)
print(">>>")

i = 0
# code_tokens = []
for line in bounding_boxes:
    j = 0
    # line = []
    for x,y,w,h in line:
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
        label = str(i) + "_" + str(j)
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        box = image[y:y+h, x:x+w]
        box_path = 'line_{}_position_{}.jpg'.format(i, j)
        cv2.imwrite(box_path, box)
        # token = get_prediction(box_path)
        # line.append(token)
        j += 1
    # code_tokens.append(line)
    i += 1

# i = 0
# for x,y,w,h in bounding_boxes:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#     cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#     i += 1

# for x,y,w,h in bounding_boxes:
#     print(x,y,w,h)
# bounding_boxes = sort_bounding_boxes(bounding_boxes)
# for x,y,w,h in bounding_boxes:
#     print(x,y,w,h)
# i = 0
# max_height = 0
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     max_height = max(max_height, h)
# nearest = max_height * 1.4
# print(cnts)
# cnts = np.array(cnts)
# cnts.sort(key=lambda r: [int(nearest * round(float(r[1]) / nearest)), r[0]])
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
#     cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
#     cv2.putText(image, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
#     print(x, y)
#     i += 1

cv2.imshow('thresh', thresh)
cv2.imshow('opening', opening)
cv2.imshow('dilate', dilate)
cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()