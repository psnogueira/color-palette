# Requirements
from model import BiSeNet
import torch
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from colorthief import ColorThief

#(True/False) to show/hide DEBUG messages
DEBUG = False

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
    
    # Definition of values
    pro = cnt / total           # The hair percentage from the face
    SB = int(SB / cnt)          # Average BLUE  hair color value
    SG = int(SG / cnt)          # Average GREEN hair color value
    SR = int(SR / cnt)          # Average RED   hair color value
    brigh = brigh / cnt / 3     # The hair brightness

    # Show RGB, hair percentage and brightness values
    if(DEBUG):
        print(f"R{SR} G{SG} B{SB}")
        print(f"pro {pro:,.4f} bright {brigh:,.4f}")
        
    return brigh

# The evaluate function uses the pre-trained BiSeNet model loaded from a file
# and passes an input image to it for parsing using the vis_parsing_maps function.    
def evaluate(cp='model/model.pth', input_path='4.jpg'):

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
        # image = img.resize((512, 512), Image.ANTIALIAS)
        # image = img.resize((512, 512), Image.NEAREST)
        # image = img.resize((512, 512), Image.LANCZOS)
        # image = img.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cpu()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        return(vis_parsing_maps(image, origin, parsing, stride=1))

if __name__ == "__main__":
    # Brightness value
    brightness = evaluate(input_path='files/4.JPG')
    print(f"Brightness: {brightness}")
   
if(DEBUG): 
    print("Fim")
