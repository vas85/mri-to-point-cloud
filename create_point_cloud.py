#!/usr/bin/env python3
import errno
import os
import sys
import struct
from PIL import Image
from PIL import GifImagePlugin
import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.ndimage

# constants
IMAGE_SCALE_FACTOR = 4
Z_STEP = 0.025

def getIfromRGB(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    RGBint = (red<<16) + (green<<8) + blue
    return RGBint

def dump_points(points, depth):
    calcPoints = []
    height = points.shape[0]
    width = points.shape[1]

    # setup rotation vector
    rotation_axis = np.array([1, 0, 0])
    rotation_vector = np.radians(90) * rotation_axis
    rotation = R.from_rotvec(rotation_vector)

    # write out the points
    for row in range(0, height):
        for col in range(0, width):
            xNorm = (col / width) - 0.5
            yNorm = (row / height) - 0.5
            pointsRgb = points[row, col]
            if pointsRgb[0] > 0 or pointsRgb[1] > 0 or pointsRgb[2] > 0:
                rotatedPoint = rotation.apply([xNorm, yNorm, depth])
                calcPoints.append([rotatedPoint[0] * 0.5, rotatedPoint[1] * 0.5, rotatedPoint[2] * 0.5, getIfromRGB(pointsRgb)])
    return calcPoints

def write_points(points, fileName):
    file = open(fileName, "wb")
    headers = [
        "VERSION .5",
        "FIELDS x y z rgb",
        "SIZE 4 4 4 4",
        "TYPE F F F",
        "HEIGHT 1",
        "POINTS %i" % (len(sum(points, []))),
        "DATA binary"
    ]
    for header in headers:
        file.write(("%s\n" % (header)).encode('ASCII'))
    for pointRow in points:
        for point in pointRow:
            file.write(struct.pack("f", point[0]))
            file.write(struct.pack("f", point[1]))
            file.write(struct.pack("f", point[2]))
            file.write(struct.pack("I", point[3]))
    file.close()

def main():
    file_name = "./image/img.gif"
    output_file_name = "pointcloud.pcd"
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
    if len(sys.argv) > 2:
        output_file_name = sys.argv[2]
    if not os.path.isfile(file_name):
        print("Usage: ./create_point_cloud.py <gif file> [output path]")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file_name)

    print("Reading", file_name)
    imageObject = Image.open(file_name)
    zDepth = 0.0
    frameCount = 0
    points = []
    for frame in range(0, imageObject.n_frames):
        imageObject.seek(frame)
        img = imageObject.convert('RGB')
        img = img.resize((int(imageObject.width / IMAGE_SCALE_FACTOR), int(imageObject.height / IMAGE_SCALE_FACTOR)), Image.NEAREST)
        points.append(dump_points(np.array(img), zDepth))
        zDepth += Z_STEP
        frameCount += 1
        sys.stdout.write("\r{}%".format(round(frameCount * 100 / imageObject.n_frames), 2))
        sys.stdout.flush()
    write_points(points, output_file_name)

if __name__ == "__main__":
    main()