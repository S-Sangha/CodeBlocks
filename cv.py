import cv2
import pytesseract
from PIL import Image

# Install pytesseract and pillow
# pip install pytesseract pillow

# Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Read the image using OpenCV
image_path = '/home/abdul/IC_Hack/CodeBlocks/else.jpg'
img = cv2.imread(image_path)
# img = Image.open(image_path)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to get a binary image
_, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY_INV)

# Use pytesseract to perform OCR on the binary image
custom_config = r'--oem 3 --psm 6 outputbase digits'
text = pytesseract.image_to_string(binary_img, config=custom_config)

# Get bounding boxes for each detected word
boxes = pytesseract.image_to_boxes(binary_img, config=custom_config)

# Draw bounding boxes on the original image
for box in boxes.splitlines():
    b = box.split()
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img, (x, img.shape[0] - y), (w, img.shape[0] - h), (0, 255, 0), 2)

# Display the result
cv2.imshow('Detected Words', img)
cv2.waitKey(0)
cv2.destroyAllWindows()