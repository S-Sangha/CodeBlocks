import cv2
import pytesseract
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

def find_bounding_boxes(image):
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
    bounding_boxes = list(filter(lambda b : b[2] * b[3] > 3000, bounding_boxes))
    bounding_boxes = sort_bounding_boxes(bounding_boxes)
    return bounding_boxes

def normalise_token(text):
    # labels = ["else","if","while","print","repeat","add","sub","mult","div","=",
    #           "and","or","not","divisible","1","2","3","4","5","6","7","8","9","0","do","end","gt","lt"]

    # NEED TO ADD DIV
    token = text.strip().lower().replace('.', '')
    match token:
        case "+":
            return "add"
        case "-":
            return "sub"
        case "x":
            return "mult"
        case ">":
            return "gt"  
        case "<":
            return "lt"       
        case "{":
            return "1"    
        case "÷":
            return "div"
        # If an exact match is not confirmed, this last case will be used if provided
        case _:
            if len(token) == 0:
                return "="
            if len(token) < 3 and token[0] == '+':
                return normalise_token(token[0])
            if token[0] == 'p':
                return "print"
            if token[0] == 'r':
                return "repeat"
            if token[0] == 'w':
                return "while"
            if len(token) > 6:
                return "divisible"
            if len(token) == 2:
                return "if"
            return token

def image_to_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    # erode = cv2.erode(opening, np.ones((5,5), np.uint8), iterations=1)
    # invert = 255 - erode
    invert = 255 - opening

    # Perform text extraction
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
    return normalise_token(data)

def extract_code_from_boxes(image, bounding_boxes):
    code_tokens = []
    for l in bounding_boxes:
        line = []
        for x,y,w,h in l:
            box = image[y:y+h, x:x+w]
            box_path = 'tmp.png'
            cv2.imwrite(box_path, box)
            token = image_to_text(box_path)
            line.append(token)
        code_tokens.append(line)
    return code_tokens

def show_bounding_boxes(image, bounding_boxes):
    i=1
    for line in bounding_boxes:
        j=1
        for x,y,w,h in line:
            box = image[y:y+h, x:x+w]
            print(w * h)
            box_path = 'tmp.png'
            cv2.imwrite(box_path, box)
            text = image_to_text(box_path)
            token = normalise_token(text)
            print(f"{text} -> {token}")
            cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
            cv2.putText(image, token, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            j += 1
        i += 1
    cv2.imshow('image', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
def analyse_image(image):
    bounding_boxes = find_bounding_boxes(image)
    code_tokens = extract_code_from_boxes(image, bounding_boxes)
    print(">>> Code tokens are:")
    print(code_tokens)
    return code_tokens

def annotate_image(image):
    bounding_boxes = find_bounding_boxes(image)
    show_bounding_boxes(image, bounding_boxes)


if __name__ == "__main__":
    image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/simple_program_examples/eg11.jpg')
    # image = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/flash.jpg')
    annotate_image(image)
    analyse_image(image)