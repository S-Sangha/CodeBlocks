import cv2
import matplotlib.pyplot as plt


import pytesseract

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
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = image
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
    return bounding_boxes

def normalise_token(text):
    return text.strip().lower()

def image_to_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Morph open to remove noise and invert image
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    invert = 255 - opening

    # Perform text extraction
    data = pytesseract.image_to_string(invert, lang='eng', config='--psm 8')
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
    
def analyse_image(image):
    bounding_boxes = find_bounding_boxes(image)
    code_tokens = extract_code_from_boxes(image, bounding_boxes)
    print(code_tokens)
    return code_tokens


code = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/true_2.jpg')
target =  cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/true.jpg')

# analyse_image(image)

def get_code_true():

    target =  cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/images/repeat_20.jpg',cv2.IMREAD_GRAYSCALE)
    program = []

    bounding_boxes = find_bounding_boxes(target)

    for l in bounding_boxes:
        for x,y,w,h in l:
            box = target[y:y+h, x:x+w]
            box_path = 'tmp.jpg'
            cv2.imwrite(box_path, box)
            candidate = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/tmp.jpg',cv2.IMREAD_GRAYSCALE)
            code = compute_match(candidate)
        
            if code != "":
                program.append(code)

    print(program)
            


def compute_match(source):
    img_list = ["/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/repeat.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/2.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/0.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/if.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/divisible.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/3.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/then.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/print.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/end.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/add.jpg",
                "/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/1.jpg",
    ]
    labels = ["repeat","2","0","if","divisibile","3","then","print","end","+","1"]
    orb = cv2.ORB_create()

    score = 0
    res = -1

    for i in range(len(img_list)):
        img = cv2.imread(img_list[i],cv2.IMREAD_GRAYSCALE)
        print(img == None)
        kp1, des1 = orb.detectAndCompute(source,None)
        kp2, des2 = orb.detectAndCompute(img,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img_score = 0
        
        # if len(matches) < 10:
        #     continue
        good_matches = [match for match in matches if match.distance < 0.75 * matches[0].distance]

        if len(good_matches) > 0.7:
            print("Images match!")
            print(labels[res])
            print("-------")
            return labels[res]
        else:
            print("Images do not match.")
            return ""


img1 = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/ground_truth/print.jpg',cv2.IMREAD_GRAYSCALE)          # queryImage
img2 = cv2.imread('/Users/sksangha/year4/ichack24/CodeBlocks/repeat_2.jpg',cv2.IMREAD_GRAYSCALE)         # queryImage
print(img2.shape)
img1 = cv2.resize(img1,(244,244))
img2 = cv2.resize(img2,(244,244))



# Initiate ORB detector 
orb = cv2.ORB_create()
# find the keypoints and descriptors with ORB

# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.

for i in range(10):
    print(f"Distance for match {i + 1}: {matches[i].distance}")
    
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:100],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()


# get_code_true()