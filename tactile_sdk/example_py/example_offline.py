from tactile_sdk.sharpa.tactile import InferOffline
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


cv2.namedWindow('initial', cv2.WINDOW_NORMAL)
cv2.namedWindow('realtime', cv2.WINDOW_NORMAL)
cv2.namedWindow('deform', cv2.WINDOW_NORMAL)


def readImages(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    return img

inferOffline = InferOffline("config/models/DEV2pin_400_de7d860c.onnx")



initial_file='./data/initial.bmp'
realtime_file='./data/realtime.bmp' 

realtime_img=readImages(realtime_file)
initial_img=readImages(initial_file)
frame=inferOffline.infer(initial_img, realtime_img, 240*320)
f6, deform, contacts = None, None, None
if frame['content'].get('DEFORM') is not None: deform = frame['content']['DEFORM'].squeeze()
if frame['content'].get('F6') is not None: f6 = frame['content']['F6'].squeeze()
if frame['content'].get('CONTACT_POINT') is not None: contacts = frame['content']['CONTACT_POINT'].squeeze().reshape(-1,3)


cv2.imshow("realtime", realtime_img)
cv2.setWindowTitle("realtime", f"realtime: {os.path.basename(realtime_file)}")
cv2.imshow("initial", initial_img)
cv2.setWindowTitle("initial", f"initial: {os.path.basename(initial_file)}")

if deform is None: 
    pass 
deform_img = cv2.cvtColor(deform, cv2.COLOR_GRAY2BGR)
z = 20
for var in f6:
    cv2.putText(deform_img, f'{var:.3f}',
            (5, z), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    z += 20

if contacts is not None:
    for contact in contacts:
        x, y, z = contact
        if z < 5: continue
        x=int(x)
        y=int(y)
        cv2.circle(deform_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(deform_img, f'{z:.2f}',(x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255.0), 2)
cv2.imshow(f'deform', deform_img)
cv2.waitKey()
cv2.destroyAllWindows()



