# Required modules
import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorthief import ColorThief

# Load trained module haarcascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video
capture = cv2.VideoCapture(0)
if not capture.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

while True:
    ret, frame = capture.read()

    if not ret:
        break

    # Face detection
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Capture face
        face_roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 165, 0), 1)

    cv2.imshow("Reconhecimento Facial", frame)

    # Press 'c' to Capture face image
    key = cv2.waitKey(1)
    if key == ord('c') and len(faces) > 0:
        cv2.imwrite("captured_face.jpg", face_roi)
        print("Foto capturada!")
        break
    elif key == 27:  # 'ESC'
        break

capture.release()
cv2.destroyAllWindows()

img_source = "captured_face.jpg"
img_destination = "ycrcb_captured_face.png"
img_destination2 = "rosto_com_paleta.png"

min_YCrCb = np.array([80, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

# Get pointer to video frames from primary device
image = cv2.imread(img_source)
imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

# Save image as .png
cv2.imwrite("foto.png", np.hstack([image,skinYCrCb])) # 2 images
cv2.imwrite(img_destination, skinYCrCb) # 1 image

# Get predominant color from image
ct = ColorThief("foto.png")

dominant_color = ct.get_color(quality=1)

palette = ct.get_palette(color_count=5)

dominant_color = palette[0]
if ((palette[0][0] < 40 and palette[0][1] < 40) or (palette[0][1] < 40 and palette[0][2] < 40) or(palette[0][0] < 40 and palette[0][2] < 40)):
    dominant_color = palette[1]

# Show rgb value # 
# print(dominant_color)

# plt.imshow([[dominant_color]])
# plt.show()
# for i in range(5):
    # print(palette[i])

# Getting color warmth
r = dominant_color[0]
g = dominant_color[1]
b = dominant_color[2]
quente = False

imagem_rosto = cv2.imread(img_source)
# print('Original Dimensions1 : ', imagem_rosto.shape)
width = int(imagem_rosto.shape[1])
height = int(imagem_rosto.shape[0])
dim = (width, height)

if(((g - b) > 23)): 
    verao = cv2.imread("paleta_verao.png")
    resized = cv2.resize(verao, dim, interpolation = cv2.INTER_AREA)
    quente = True
    print("o tom de pele é quente")
    cv2.imwrite(img_destination2 ,np.hstack([imagem_rosto,resized]))
    # cv2.imwrite(img_destination2, resized)
    # print('Original Dimensions2 : ', resized.shape)
    foto_paleta = cv2.imread(img_destination2)
    cv2.imshow("QUENTE", foto_paleta)
else:
    primavera = cv2.imread("paleta_primavera.png")
    resized = cv2.resize(primavera, dim, interpolation = cv2.INTER_AREA)
    print("o tom de pele é frio")
    cv2.imwrite(img_destination2 , np.hstack([imagem_rosto,resized]))
    # cv2.imwrite(img_destination2, resized)
    # print('Original Dimensions2 : ', resized.shape)
    foto_paleta = cv2.imread(img_destination2)
    cv2.imshow("FRIA", foto_paleta)
    
cv2.waitKey(0)
cv2.destroyAllWindows()

# plt.imshow([[palette[i] for i in range(5)]])
# plt.show()