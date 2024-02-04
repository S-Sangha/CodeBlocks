# import cv2
# import numpy as np

# # Load the pre-trained MobileNet SSD model
# net = cv2.dnn.readNetFromTensorflow('ssd_mobilenet_v2_coco.pb', 'ssd_mobilenet_v2_coco.pbtxt')

# # Read the input image
# image_path = 'images/blocks.jpg'
# img = cv2.imread(image_path)
# height, width = img.shape[:2]

# # Preprocess the image for object detection
# blob = cv2.dnn.blobFromImage(img, 0.007843, (300, 300), 127.5)

# # Set the input to the network
# net.setInput(blob)

# # Run forward pass to get the detection results
# detections = net.forward()

# # Loop over the detections and draw bounding boxes
# for i in range(detections.shape[2]):
#     confidence = detections[0, 0, i, 2]
#     if confidence > 0.5:  # You can adjust the confidence threshold
#         class_id = int(detections[0, 0, i, 1])
#         box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
#         (startX, startY, endX, endY) = box.astype("int")

#         # Draw the bounding box and label on the image
#         label = f"Class: {class_id}, Confidence: {confidence:.2f}"
#         cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         y = startY - 15 if startY - 15 > 15 else startY + 15
#         cv2.putText(img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# # Display the result
# cv2.imshow('Object Detection', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
