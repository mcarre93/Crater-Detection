import cv2 as cv
import time
from datetime import timedelta
import os
import pandas as pd
from helper import calculateDistance, delete_duplicate_keep_best, draw_craters_rectangles

# the raw data to process for duplicate removal
csv_path = 'results/crater_24_cnn.csv'

# the image for drawing rectangles
img_path = os.path.join('crater_data', 'images', 'tile3_24.pgm')

data = pd.read_csv(csv_path, header=None)

start_time = time.time()

# first pass, remove duplicates for points of same window size
df1 = {}
for ws in data[2].unique():
    df1[ws] = data[ (data[3] > 0.75) & (data[2] == ws) ] # take only 75% or higher confidence
    df1[ws] = delete_duplicate_keep_best(df1[ws])

# Start merging process
# We will add points of greatest size first
# then merge with the next smaller size and remove duplicates
# Do this until the smallest window size has been included

merge = pd.DataFrame()
for ws in reversed(sorted(df1.keys())):
    merge = pd.concat([merge, df1[ws]])
    old_size = len(merge)
    merge = delete_duplicate_keep_best(merge) # we can tweak ws for eliminations
    new_size = len(merge)
    print("Processed window size", ws, ", considered", old_size, "points, returned", new_size, "points")

img = draw_craters_rectangles(img_path, merge, show_probs=False)
cv.imwrite("%s.jpg" % (csv_path.split('.')[0]), img, [int(cv.IMWRITE_JPEG_QUALITY), 100])
merge[[0,1,2]].to_csv("%s_noduplicates.csv" % (csv_path.split('.')[0]), header=False, index=False)

end_time = time.time()
time_dif = end_time - start_time
print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))