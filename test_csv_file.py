__author__ = 'navid'
import csv
from urllib import request


import numpy as np
import pandas as pd
from PIL import Image

from heatmappy import Heatmapper
fname = [r"D:\Dropbox (Lumen)\Team Lumen\Data & Modelling\ctt-evaluation-analysis\r_project\del\aoi_cv_study15_1to2_SigmoidTwoNearestNeighbours_2017-10-05 04-PM.csv",

         r"D:\Dropbox (Lumen)\Team Lumen\Data & Modelling\ctt-evaluation-analysis\r_project\fordavid\gaze_points_with_variance-study9\gaze_cv_study9_1to300_SigmoidTwoNearestNeighbours_2017-10-07 04-AM.csv"]
fname = fname[1]

print("processing {}".format(fname))

def ellipse_np_example(example_img, example_points):
    heatmapper = Heatmapper(colours='default', point_strength=1, grey_heatmapper='PILNPAreaDiscountEllipse')
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_ellipse.png")

def color_example(example_img, example_points):
    heatmapper = Heatmapper(colours='default')
    img = heatmapper.heatmap_on_img(example_points, example_img)
    img.show()
    img.save("out_color.png")

def prepare_gaze(x, y, xvar, yvar, xycovar, mean_precision):

    cov = np.array([[xvar, xycovar], [xycovar, yvar]])
    nstd = 2

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)


    # find precision level
    # if mean_precision > 300:
    #     width = 0
    #     height = 0

    return int(x), int(y), int(width), int(height), int(theta)

df = pd.read_csv(fname, sep=',', header=0)

for g in df.groupby('stimuli_id'):
    stimuli_id = g[0]



    img_fname = '{}.jpg'.format(stimuli_id)
    print(stimuli_id)
    stimulus_src = g[1].stimulus_src.head(1).values[0]
    print(stimulus_src)
    stim_df = g[1]

    if stimuli_id != 22:
        continue

    with open(img_fname, 'wb') as f:
        f.write(request.urlopen(stimulus_src).read())

    # np, discount area
    example_img = Image.open(img_fname)
    compiled_gazes = (prepare_gaze(row['gazes.x'],
                                   row['gazes.y'], row['calib_cluster_variance_x'],
                                   row['calib_cluster_variance_y'], row['calib_cluster_variance_xy'],
                                   [row['precision_px_validator-1'],
                                   row['precision_px_validator-2'],
                                   row['precision_px_validator-3'],
                                   row['precision_px_validator-4'],
                                   row['precision_px_validator-5']])
                      for index, row in stim_df.iterrows())
    ellipse_np_example(example_img, compiled_gazes)

    ## np, discount area
    # example_img = Image.open(img_fname)
    # compiled_gazes = ((row['gazes.x'], row['gazes.y']) for index, row in stim_df.iterrows())
    # color_example(example_img, compiled_gazes)






# with open() as csvfile:
#     data = csv.DictReader(csvfile)
#
#     for row in data:
#         print(row)
#
# example_points = [(100, 20), (120, 25), (200, 50), (60, 300), (170, 250)]
# example_img_path = 'heatmappy/assets/cat.jpg'
# example_img = Image.open(example_img_path)
# heatmapper = Heatmapper()
# heatmap = heatmapper.heatmap_on_img(example_points, example_img)
# heatmap.show()