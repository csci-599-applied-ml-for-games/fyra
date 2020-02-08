import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import argparse

## use to plot trajectory tracking error for test_controller.py

TX=13 #target x
TY=14
TZ=15
CX=1 #current x
CY=2
CZ=3

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("data", type=str, help="CSV file data")
  parser.add_argument("--threeD", action='store_true', help="3D plot")
  args = parser.parse_args()

  data = np.loadtxt(args.data, delimiter=",")

  target = data[:,TX:TZ+1]
  current = data[:,CX:CZ+1]
  error = np.linalg.norm(target - current, axis=1)
  # print(error)

  # new figure
  plt.figure(0)

  if args.threeD:
    plt.subplot(1, 1, 1, projection='3d')
    plt.plot(data[:,CX], data[:,CY], data[:,CZ], label='current')
    plt.plot(data[:,TX], data[:,TY], data[:,TZ], label='target')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)
  else:
    # X, Y, Z
    plt.subplot(2, 1, 1)
    plt.plot(current[:,0], current[:,1], '-', label='current')
    plt.plot(target[:,0], target[:,1], '-', label='target')
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

    plt.subplot(2, 1, 2)
    plt.plot(current[:,2], '-', label='current')
    plt.plot(target[:,2], '-', label='target')
    plt.xlabel('dataset/time')
    plt.ylabel('Z [m]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

  print("avg. error: {:.3f} m, stddev: {:.3f}".format(
    np.mean(error),
    np.std(error)))

  plt.show()

