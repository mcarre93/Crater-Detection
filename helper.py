import numpy as np
import math
import pandas as pd
import cv2 as cv
import os

def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def calculateDistance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def delete_duplicate_keep_best(dataframe):
    # Note this method now takes a dataframe as input
    
    if len(dataframe) < 2:
        # nothing to do
        return dataframe

    Crater_data = dataframe
    # extract axes
    x = Crater_data[0].values.tolist()
    y = Crater_data[1].values.tolist()
    s = Crater_data[2].values.tolist()
    p = Crater_data[3].values.tolist()
    Points = []
    while len(x) > 0:
        # a group is a set of similar points
        Group_x = [x[0]]
        Group_y = [y[0]]
        Group_s = [s[0]]
        Group_p = [p[0]]
        index = [0]
        for i in range(1,len(x)):
            d_current = calculateDistance(x[0],y[0],x[i],y[i])

            # accept in group only if 
            d = min(s[0], s[i])
            if d_current <= d and s[i] < 2*d:
                Group_x.append(x[i])
                Group_y.append(y[i])
                Group_s.append(s[i])
                Group_p.append(p[i])
                index.append(i)
        # after group is defined, extract its elements from list
        x = list(np.delete(x,index))
        y = list(np.delete(y,index))
        s = list(np.delete(s,index))
        p = list(np.delete(p,index))
        Points.append([Group_x,Group_y,Group_s, Group_p])

    # now reduce groups
    center_size = []
    for i, (Xs, Ys, Ss, Ps) in enumerate(Points):
        # we take the point with best prediction confidence
        best_index = np.argmax(Ps)
        x_center = Xs[best_index]
        y_center = Ys[best_index]
        size = Ss[best_index]
        prob = Ps[best_index]
        center_size += [[x_center,y_center,size, prob]]

    return pd.DataFrame(center_size)

def draw_craters_rectangles(img_path, dataframe, show_probs=True):
    img_filename = os.path.basename(img_path)
    img = cv.imread(img_path, 1)
    img = cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)

    color = (0, 255, 0)
    font = cv.FONT_HERSHEY_SIMPLEX
    
    for index, row in dataframe.iterrows():
        winS = int(row[2])
        half_winS = int(winS/2)
        x = int(row[0] - half_winS)
        y = int(row[1] - half_winS)
        # if we want to see where is processed.
        cv.rectangle(img, (x, y), (x + winS, y + winS), color, 2)
        if show_probs:
            cv.putText(img, "%f" % row[3], (x, y-5), font, 0.6, color, 2)

    return img