import os
import numpy as np
from utils.general import *

countries = {"China_Drone" : [0,0,0,0], "Chine_MotorBike": [0,0,0,0],"Czech": [0,0,0,0],"India": [0,0,0,0],"Japan": [0,0,0,0],"Norway": [0,0,0,0],"United_States": [0,0,0,0]}
aspect_ratios_sum = np.array([0,0,0,0])
total_label_sum = np.array([0,0,0,0])
for key, item in countries:
    directory = f"./data/RDD2022/{key}/train/labels/"
    for filename in os.listdir(directory):
        with open(filename, 'r') as f:
            for line in f:
                x = line.split()
                type = int(x[0])
                
                [xmin, ymin, xmax, ymax] = int(x[1:5])
                [x,y,w,h] = xyxy2xywh([xmin,ymin,xmax,ymax])
                ar = w/h

                aspect_ratios_sum[type] += ar
                item[type] += 1
                total_label_sum[type] += 0

aspect_ratios = aspect_ratios_sum/total_label_sum
print(aspect_ratios)
print(countries)

    
