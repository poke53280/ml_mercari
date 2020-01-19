
import time
import torch
import cv2
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm.notebook import tqdm

from facenet_pytorch import MTCNN

import pandas as pd





# Load face detector
mtcnn = MTCNN(margin=14, keep_all=True, factor=0.5, device=device, post_process=False).eval()


def process(filename):
    
    v_cap = cv2.VideoCapture(str(filename))
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    v_width = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    v_height = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    outfile = filename.parent / f"{filename.stem}_boxed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*'FMP4')    

    dim = (v_width, v_height)

    video_tracked = cv2.VideoWriter(str(outfile), fourcc, 25.0, dim)

    # Loop through frames
    faces = []
    probs = []

    l_frame = []

    for j in range(v_len):

        print (j)

        success = v_cap.grab()
        success, frame = v_cap.retrieve()
        if not success:
            print("Error")
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        l_frame.append(frame)

        frame = Image.fromarray(frame)

        boxes, prob  = mtcnn.detect(frame)

        faces.append(boxes)
        probs.append(prob)

        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
        
        video_tracked.write(cv2.cvtColor(np.array(frame_draw), cv2.COLOR_RGB2BGR))

    v_cap.release()
    video_tracked.release()

    acFrame = np.stack(l_frame)

    nFaces = [len(x) for x in faces]

    nFace_target = np.median(nFaces)
    nFace_max    = np.max(nFaces)

    d = {}

    for iFace in range (nFace_max):

        l_face = []

        for x in faces:
            if len(x) >= iFace + 1:
                l_face.append(list(x[iFace]))
            else:
                l_face.append([np.nan,np.nan, np.nan, np.nan])
        """c"""

        d[iFace] = l_face

    l_S = [pd.Series(d[iFace], name = f"face{iFace}") for iFace in range (nFace_max)]

    df = pd.DataFrame(l_S).T

    df = df.assign(N = nFaces)

    l_face = [x for x in list(df.columns) if x.startswith('face')]

    l_S = []

    for face in l_face:
        s = df[face]

        for i in range(4):
            x = s.map(lambda x:x[i])
            x.name = f"{face}_p{i}"
            l_S.append(x)

    df_S = pd.DataFrame(l_S).T
    df = pd.concat([df, df_S], axis = 1)

    l_S = []

    for iFace in range (nFace_max):
        w = df.apply(lambda x: x[f"face{iFace}_p2"] - x[f"face{iFace}_p0"], axis = 1)
        w.name = f"w{iFace}"
        h = df.apply(lambda x: x[f"face{iFace}_p3"] - x[f"face{iFace}_p1"], axis = 1)
        h.name = f"h{iFace}"

        cx = df[f"face{iFace}_p0"] + w/2
        cx.name = f"cw{iFace}"

        cy = df[f"face{iFace}_p1"] + h/2
        cy.name = f"ch{iFace}"

        l_S.append(w)
        l_S.append(h)
        l_S.append(cx)
        l_S.append(cy)



    df_S = pd.DataFrame(l_S).T
    df = pd.concat([df, df_S], axis = 1)


    l_drop = [x for x in list(df.columns) if x.startswith("face")]

    df = df.drop(l_drop, axis = 1)

    # Face 0

    m = np.abs(df.cw0-df.cw0.mean())<=(3*df.cw0.std())

    m.sum()/ m.shape[0]
    # All OK

    




    cw0.mean()

    cw0cw0
    cw0


    nFace_target




    return faces


k = process()

for x in k:
    if len(x) != 2:
        print (x)


with torch.no_grad():
    faces = detection_pipeline(filename)
   





