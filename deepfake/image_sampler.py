
import numpy as np


#################################################################################
#
#   bresenham
#
#   https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python

def bresenham(x0, y0, x1, y1):

    l = []
    
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            l.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            l.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    l.append((x, y))

    return l


width = 1024
height = 728

x0 = 5 + np.random.choice(width - 10)
y0 = 5 + np.random.choice(height - 10)

rAngle = np.random.uniform(- np.pi, np.pi)

# Float version. Find orthogonal vector. Expand to thickness.


x1 = int (x0 + 10.0 * np.cos(rAngle))
y1 = int (y0 + 10.0 * np.sin(rAngle))

# Location metrics

rY   = y0/height
rX   = x0/width

l = bresenham(x0, y0, x1, y1)[:5]


aX = np.array([x[0] for x in l])
aY = np.array([x[1] for x in l])




