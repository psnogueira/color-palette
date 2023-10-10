# Required modules
from model import BiSeNet
import torch
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorthief import ColorThief
from tkinter import *
from PIL import Image, ImageTk

#(True/False) to show/hide DEBUG messages
DEBUG = False

# Paths
img_source = 'files_temp/captured_face.jpg'
img_color_destination = "files_temp/foto.png"
img_destination = 'files_temp/ycrcb_captured_face.png'
img_destination2 = 'files_temp/rosto_com_paleta.png'

# Load trained module haarcascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# The similar Function calculates the similarity of two colors 
# by comparing their Green (G), Blue (B), and Red (R) values.
def similar(G1,B1,R1,G2,B2,R2):
    ar=[]
    if G2 > 30:
        ar.append(1000.*G1/G2)
    if B2 > 30:
        ar.append(1000.*B1/B2)
    if R2 > 30:
        ar.append(1000.*R1/R2)
    if len(ar) < 1:
        return False
    if min(ar) == 0:
        return False
    br = max(R1,G1,B1) / max(G2,B2,R2)
    return max(ar) / min(ar) < 1.55 and br > 0.7 and br < 1.4

# The vis_parsing_maps function takes 'input image', 'origin image', 'parsing annotation'
# and 'stride', as arguments and performs facial and hair parsing.
def vis_parsing_maps(im, origin, parsing_anno, stride):

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    num_of_class = np.max(vis_parsing_anno)

    SB = 0
    SR = 0
    SG = 0
    cnt = 0
    total = 0
    brigh = 0
    FB = 0
    FR = 0
    FG = 0
    FN = 0
    for x in range(0, origin.shape[0]):
        for y in range(0, origin.shape[1]):
            _x = int(x * 512 / origin.shape[0])
            _y = int(y * 512 / origin.shape[1])
            if vis_parsing_anno[_x][_y] == 1:
                FB = FB + int(origin[x][y][0])
                FG = FG + int(origin[x][y][1])
                FR = FR + int(origin[x][y][2])
                FN = FN + 1
    FB = int(FB / FN)
    FR = int(FR / FN)
    FG = int(FG / FN)

    for x in range(0, origin.shape[0]):
        for y in range(0, origin.shape[1]):
            _x = int(x * 512 / origin.shape[0])
            _y = int(y * 512 / origin.shape[1])
            if vis_parsing_anno[_x][_y] == 17:
                OB = int(origin[x][y][0])
                OG = int(origin[x][y][1])
                OR = int(origin[x][y][2])
                if similar(OB,OG,OR,FB,FG,FR) :
                    continue
                SB = SB + OB
                SG = SG + OG
                SR = SR + OR
                cnt = cnt + 1
                brigh = brigh + OR + OG + OR
            if vis_parsing_anno[_x][_y] <= 17:
                total = total + 1
    try:
        # Definition of values
        pro = cnt / total           # The hair percentage from the face
        SB = int(SB / cnt)          # Average BLUE  hair color value
        SG = int(SG / cnt)          # Average GREEN hair color value
        SR = int(SR / cnt)          # Average RED   hair color value
        brigh = brigh / cnt / 3     # The hair brightness
    except:
        # Bald
        pro = 0                     # The hair percentage from the face
        SB = 0                      # Average BLUE  hair color value
        SG = 0                      # Average GREEN hair color value
        SR = 0                      # Average RED   hair color value
        brigh = 100                 # The hair brightness

    # Show RGB, hair percentage and brightness values
    if(DEBUG):
        print(f"R{SR} G{SG} B{SB}")
        print(f"pro {pro:,.4f} bright {brigh:,.4f}")
        
    return brigh

# The evaluate function uses the pre-trained BiSeNet model loaded from a file
# and passes an input image to it for parsing using the vis_parsing_maps function.    
def evaluate(cp='model/model.pth', input_path=img_source):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cpu()
    save_pth = osp.join('', cp)
    net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        img = Image.open(input_path)
        origin = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        image = img.resize((512,512))
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cpu()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        return(vis_parsing_maps(image, origin, parsing, stride=1))

if __name__ == "__main__":
    # Start video
    #
    # If more than 1 camera is connected,
    # change the camera index in "cv2.VideoCapture(0)"
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
            cv2.imwrite(img_source, face_roi)
            print("Foto capturada!")
            break
        elif key == 27:  # 'ESC'
            break

    capture.release()
    cv2.destroyAllWindows()

    min_YCrCb = np.array([80, 133, 77], np.uint8)
    max_YCrCb = np.array([235, 173, 127], np.uint8)

    # Get pointer to video frames from primary device
    image = cv2.imread(img_source)
    imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)

    skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)

    # Save image as .png
    cv2.imwrite(img_color_destination, np.hstack([image,skinYCrCb])) # 2 images
    cv2.imwrite(img_destination, skinYCrCb) # 1 image

    # Get predominant color from image
    ct = ColorThief(img_color_destination)

    dominant_color = ct.get_color(quality=1)

    palette = ct.get_palette(color_count=5)

    dominant_color = palette[0]
    if ((palette[0][0] < 40 and palette[0][1] < 40) or 
        (palette[0][1] < 40 and palette[0][2] < 40) or 
        (palette[0][0] < 40 and palette[0][2] < 40)):
        
        dominant_color = palette[1]

    # Getting color warmth
    r = dominant_color[0]
    g = dominant_color[1]
    b = dominant_color[2]
    quente = False

    imagem_rosto = cv2.imread(img_source)
    width = int(imagem_rosto.shape[1])
    height = int(imagem_rosto.shape[0])
    dim = (width, height)
    
    # Brightness value
    brightness = evaluate(input_path=img_source)
    if(DEBUG): print(f"Brightness: {brightness}")

    # if(((g - b) > 23)): 
    if(DEBUG):
        print(f"g{g} b{b}")
    if(((g - b) < 23)): 
        # Cool palette
        
        # if(brightness > 82):
        if(brightness > 63):
            # Light Hair
            # Summer
            
            if(DEBUG): print("Paleta Verao")
            
            verao = cv2.imread("files/paleta_verao.png")
            
            resized = cv2.resize(verao, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(img_destination2, np.hstack([imagem_rosto,resized]))
            foto_paleta = cv2.imread(img_destination2)

            # Create window
            window = Tk()

            # Window title
            window.title("Paleta de Cores")

            # Logo photo
            path = "files/logo.png"
            load = Image.open(path)
            render = ImageTk.PhotoImage(load)
            window.iconphoto(False, render)

            img = PhotoImage(file="files/produtos_verao.png")
                        
            img_paleta = Image.open(img_destination2)
            img_paleta = img_paleta.resize((768,384))
            photo_paleta = ImageTk.PhotoImage(img_paleta)

            label = Label(image=photo_paleta).pack()
            label = Label(image=img).pack()

            window.mainloop()
            
        else:
            # Dark Hair
            # Winter
            
            if(DEBUG): print("Paleta Inverno")
            
            inverno = cv2.imread("files/paleta_inverno.png")
            
            resized = cv2.resize(inverno, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(img_destination2, np.hstack([imagem_rosto,resized]))
            foto_paleta = cv2.imread(img_destination2)

            # Create window
            window = Tk()

            # Window title
            window.title("Paleta de Cores")

            # Logo photo
            path = "files/logo.png"
            load = Image.open(path)
            render = ImageTk.PhotoImage(load)
            window.iconphoto(False, render)

            img = PhotoImage(file="files/produtos_inverno.png")
            img_paleta = Image.open(img_destination2)
            img_paleta = img_paleta.resize((768,384))
            photo_paleta = ImageTk.PhotoImage(img_paleta)

            label = Label(image=photo_paleta).pack()
            label = Label(image=img).pack()

            window.mainloop()
    else:
        # Warm palette
        
        # if(brightness > 82):
        if(brightness > 63):
            # Light Hair
            # Spring
            
            if(DEBUG): print("Paleta Primavera")
            
            primavera = cv2.imread("files/paleta_primavera.png")
            
            resized = cv2.resize(primavera, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(img_destination2, np.hstack([imagem_rosto,resized]))
            foto_paleta = cv2.imread(img_destination2)

            # Create window
            window = Tk()

            # Window title
            window.title("Paleta de Cores")

            # Logo photo
            path = "files/logo.png"
            load = Image.open(path)
            render = ImageTk.PhotoImage(load)
            window.iconphoto(False, render)

            img = PhotoImage(file="files/produtos_primavera.png")

            img_paleta = Image.open(img_destination2)
            img_paleta = img_paleta.resize((768,384))
            photo_paleta = ImageTk.PhotoImage(img_paleta)

            label = Label(image=photo_paleta).pack()
            label = Label(image=img).pack()

            window.mainloop()

        else:
            # Dark Hair
            # Autumn
            
            if(DEBUG): print("Paleta Outono")
            
            outono = cv2.imread("files/paleta_outono.png")
            
            resized = cv2.resize(outono, dim, interpolation = cv2.INTER_AREA)
            cv2.imwrite(img_destination2, np.hstack([imagem_rosto,resized]))
            foto_paleta = cv2.imread(img_destination2)

            # Create window
            window = Tk()

            # Window title
            window.title("Paleta de Cores")

            # Logo photo
            path = "files/logo.png"
            load = Image.open(path)
            render = ImageTk.PhotoImage(load)
            window.iconphoto(False, render)

            img = PhotoImage(file="files/produtos_outono.png")

            img_paleta = Image.open(img_destination2)
            img_paleta = img_paleta.resize((768,384))
            photo_paleta = ImageTk.PhotoImage(img_paleta)

            label = Label(image=photo_paleta).pack()
            label = Label(image=img).pack()

            window.mainloop()
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if(DEBUG): 
    print("Fim")
