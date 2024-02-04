import cv4
from pathlib import Path
import pytesseract
from PIL import Image

def preprocess_image(image_path):
    img = cv4.imread(image_path)
    gray = cv4.cvtColor(img, cv4.COLOR_BGR2RGB)
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold_image = cv4.threshold(gray, 0, 255, cv4.THRESH_BINARY | cv4.THRESH_OTSU)
    return threshold_image

def image_to_text(image_path: Path):
    image = Image.open(image_path)
    # preprocessed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(image)
    print(extracted_text)

if __name__ == '__main__':
    image_path = Path("images/blocks2.jpg")
    if (image_path.exists()):
        print(">>> Image path exists")
    else:
        print(">>> no")
    image_to_text(image_path)