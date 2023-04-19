import os
import numpy as np

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
#"China_Drone" : np.array([0,0,0,0]), "Chine_MotorBike": np.array([0,0,0,0]),
countries = {"Czech": np.array([0,0,0,0]),"India": np.array([0,0,0,0]),"Japan": np.array([0,0,0,0]),"Norway": np.array([0,0,0,0]),"United_States": np.array([0,0,0,0])}
aspect_ratios_sum = np.array([0,0,0,0])
total_label_sum = np.array([0,0,0,0])
for i, (key, item) in enumerate(countries.items()):
    directory = f"./data/RDD2022/{key}/train/labels/"
    for filename in os.listdir(directory):
        with open(directory + filename, 'r') as f:
            for line in f:
                x = line.split()
                type = int(x[0])
                
                [xmax, ymax, xmin, ymin] = [float(x[1]),float(x[2]),float(x[3]),float(x[4])]
                w = abs(xmax-xmin)
                h = abs(ymax-ymin)
                if (h == 0): ar = 0
                else: ar = w/h
                
                

                aspect_ratios_sum[type] += ar
                item[type] += 1
                total_label_sum[type] += 1

aspect_ratios = aspect_ratios_sum/total_label_sum
print(aspect_ratios)
print(countries)

    
