

#Texture attributes for detecting salt bodies in seismic data
#Tamir Hegazy
#∗ and Ghassan AlRegib
#Center for Energy and Geo Processing (CeGP), Georgia Institute of Technology

DATA_DIR_PORTABLE = "C:\\tgs_data\\"
DATA_DIR_BASEMENT = DATA_DIR_PORTABLE
DATA_DIR = DATA_DIR_PORTABLE


import os
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

img = cv2.imread(DATA_DIR + 'smooth_elf_post_img.jpg',0)


laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()


img = img.astype(np.float32)

img = img/255.0



def get_score(img, x0, y0, x1, y1):

    window = img[x0:x1, y0:y1]

    w_x = np.gradient(window, axis = 0)
    w_y = np.gradient(window, axis = 1)

    w_x = w_x.flatten()
    w_y = w_y.flatten()

    lr = linear_model.LinearRegression()
    lr.fit(w_x.reshape(-1,1), w_y)

    score = lr.score(w_x.reshape(-1,1),w_y)

    print(f"lr score = {score}")


    line_X = np.arange(w_x.min(), w_x.max(), step = .01)[:, np.newaxis]
    line_y = lr.predict(line_X)

    print(f"lr coef = {lr.coef_}")


    #rr = linear_model.RANSACRegressor()
    #rr.fit(w_x.reshape(-1,1), w_y)
    #line_y_ransac = rr.predict(line_X)


    plt.scatter(w_x, w_y)

    plt.plot(line_X, line_y, color='red', linewidth=2, label='Linear regressor')

    plt.Rectangle((50,50), 100, 150, facecolor = 'red', edgecolor = 'blue')


    #plt.plot(line_X, line_y_ransac, color='orange', linewidth=2, label='RANSAC regressor')

    plt.show()

    return score

"""c"""


# Non salt
1314, 574 , 1470, 707
330, 240, 490, 365
1247, 1066, 1370, 1160

# Salt
788, 922,   918, 1040
822, 657,  914, 763



x0 = 1314
y0 = 574
x1 = 1450
y1 = 627

x0 = 788
y0 = 922
x1 = 918
y1 = 1040



get_score(img, 1314, 574 , 1470, 707)
# => 0.12
get_score(img, 330, 240, 490, 365)
# => 0.0002

get_score(img, 1247, 1066, 1370, 1160)
# 0.07

x0 = 788
y0 = 922
x1 = 918
y1 = 1040

get_score(img, 788, 922,   918, 1040)

get_score(img, 822, 657,  914, 763)


train_fns = os.listdir(TRAIN_IMAGE_DIR)

X = [np.array(cv2.imread(TRAIN_IMAGE_DIR + p, cv2.IMREAD_GRAYSCALE), dtype=np.uint8) for p in tqdm(train_fns)]



from __future__ import print_function
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.image import AxesImage
import numpy as np
from numpy.random import rand

if 1:  # simple picking, lines, rectangles and text
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.set_title('click on points, rectangles or text', picker=True)
    ax1.set_ylabel('ylabel', picker=True, bbox=dict(facecolor='red'))
    line, = ax1.plot(rand(100), 'o', picker=5)  # 5 points tolerance

    # pick the rectangle
    bars = ax2.bar(range(10), rand(10), picker=True)
    for label in ax2.get_xticklabels():  # make the xtick labels pickable
        label.set_picker(True)

    def onpick1(event):
        if isinstance(event.artist, Line2D):
            thisline = event.artist
            xdata = thisline.get_xdata()
            ydata = thisline.get_ydata()
            ind = event.ind
            print('onpick1 line:', zip(np.take(xdata, ind), np.take(ydata, ind)))
        elif isinstance(event.artist, Rectangle):
            patch = event.artist
            print('onpick1 patch:', patch.get_path())
        elif isinstance(event.artist, Text):
            text = event.artist
            print('onpick1 text:', text.get_text())

    fig.canvas.mpl_connect('pick_event', onpick1)

if 1:  # picking with a custom hit test function
    # you can define custom pickers by setting picker to a callable
    # function.  The function has the signature
    #
    #  hit, props = func(artist, mouseevent)
    #
    # to determine the hit test.  if the mouse event is over the artist,
    # return hit=True and props is a dictionary of
    # properties you want added to the PickEvent attributes

    def line_picker(line, mouseevent):
        """
        find the points within a certain distance from the mouseclick in
        data coords and attach some extra attributes, pickx and picky
        which are the data points that were picked
        """
        if mouseevent.xdata is None:
            return False, dict()
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        maxd = 0.05
        d = np.sqrt((xdata - mouseevent.xdata)**2. + (ydata - mouseevent.ydata)**2.)

        ind = np.nonzero(np.less_equal(d, maxd))
        if len(ind):
            pickx = np.take(xdata, ind)
            picky = np.take(ydata, ind)
            props = dict(ind=ind, pickx=pickx, picky=picky)
            return True, props
        else:
            return False, dict()

    def onpick2(event):
        print('onpick2 line:', event.pickx, event.picky)

    fig, ax = plt.subplots()
    ax.set_title('custom picker for line data')
    line, = ax.plot(rand(100), rand(100), 'o', picker=line_picker)
    fig.canvas.mpl_connect('pick_event', onpick2)


if 1:  # picking on a scatter plot (matplotlib.collections.RegularPolyCollection)

    x, y, c, s = rand(4, 100)

    def onpick3(event):
        ind = event.ind
        print('onpick3 scatter:', ind, np.take(x, ind), np.take(y, ind))

    fig, ax = plt.subplots()
    col = ax.scatter(x, y, 100*s, c, picker=True)
    #fig.savefig('pscoll.eps')
    fig.canvas.mpl_connect('pick_event', onpick3)

if 1:  # picking images (matplotlib.image.AxesImage)
    fig, ax = plt.subplots()
    im1 = ax.imshow(rand(10, 5), extent=(1, 2, 1, 2), picker=True)
    im2 = ax.imshow(rand(5, 10), extent=(3, 4, 1, 2), picker=True)
    im3 = ax.imshow(rand(20, 25), extent=(1, 2, 3, 4), picker=True)
    im4 = ax.imshow(rand(30, 12), extent=(3, 4, 3, 4), picker=True)
    ax.axis([0, 5, 0, 5])

    def onpick4(event):
        artist = event.artist
        if isinstance(artist, AxesImage):
            im = artist
            A = im.get_array()
            print('onpick4 image', A.shape)

    fig.canvas.mpl_connect('pick_event', onpick4)


plt.show()


def my_on_press(event):
    print (f'press on {event.xdata}, {event.ydata}')

q = plt.gca()

q.figure.canvas.mpl_connect('button_press_event', my_on_press)



import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Annotate(object):
    def __init__(self):
        self.ax = plt.gca()
        self.rect = Rectangle((0,0), 1, 1)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.add_patch(self.rect)
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)

    def on_press(self, event):
        print ('press')
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print ('release')
        self.x1 = event.xdata
        self.y1 = event.ydata
        self.rect.set_width(self.x1 - self.x0)
        self.rect.set_height(self.y1 - self.y0)
        self.rect.set_xy((self.x0, self.y0))
        self.ax.figure.canvas.draw()

a = Annotate()
plt.show()