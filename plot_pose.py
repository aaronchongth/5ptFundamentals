import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script for plotting pose motion from text file.")
    parser.add_argument("--text-file", type=str, default="5pt.txt",
                        help="Path to text file containing pose estimations for plotting.")
    args = parser.parse_args()

    # prepare to plot in 3D
    fig = plt.figure()
    # ax = fig.add_subplot(211, projection='3d')
    a2d = fig.add_subplot(111)

    file = open(args.text_file, 'r')
    for line in file:
        x, y, z = line.split(' ')
        # ax.scatter(float(x), float(y), float(z), c='r', marker='.')
        a2d.plot(float(x), -float(z), c='r', marker='.')

    # ax.set_xlabel('Frame left')
    # ax.set_ylabel('Frame upwards')
    # ax.set_zlabel('Frame forwards')
    plt.show()
