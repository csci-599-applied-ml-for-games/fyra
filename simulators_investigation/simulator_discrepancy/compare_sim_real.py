import matplotlib.pyplot as plt
import numpy as np
import argparse


## axis
X=1
Y=2
Z=3
QX=4
QY=5
QZ=6
QW=7
VX=8
VY=9
VZ=10
Roll=11
Pitch=12
Yaw=13
t0=14
t1=15
t2=16
t3=17

def main(sim, real):
  # plot_size = 400 # len(sim), len(real)
  plot_size = len(sim) if len(sim) < len(real) else len(real)

  sim_x_axis = np.linspace(1, plot_size, plot_size)
  real_x_axis = np.linspace(1, plot_size, plot_size)

  plt.figure(0)
  # X, Y, Z
  plt.subplot(8, 1, 1)
  plt.plot(sim_x_axis, sim[:plot_size,X], '-', label='sim X')
  plt.plot(sim_x_axis, sim[:plot_size,Y], '-', label='sim Y')
  plt.plot(sim_x_axis, sim[:plot_size,Z], '-', label='sim Z')
  plt.plot(real_x_axis, real[:plot_size,X], '-', label='real X')
  plt.plot(real_x_axis, real[:plot_size,Y], '-', label='real Y')
  plt.plot(real_x_axis, real[:plot_size,Z], '-', label='real Z')
  plt.xlabel('Time [s]')
  plt.ylabel('Position [m]')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  # VX, VY, VZ
  plt.subplot(8, 1, 2)
  plt.plot(real_x_axis, real[:plot_size,VX], '-', label='real VX')
  plt.plot(real_x_axis, real[:plot_size,VY], '-', label='real VY')
  plt.plot(real_x_axis, real[:plot_size,VZ], '-', label='real VZ')
  plt.plot(sim_x_axis, sim[:plot_size,VX], '-', label='sim VX')
  plt.plot(sim_x_axis, sim[:plot_size,VY], '-', label='sim VY')
  plt.plot(sim_x_axis, sim[:plot_size,VZ], '-', label='sim VZ')
  plt.xlabel('Time [s]')
  plt.ylabel('Velocity [m/s]')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  # Roll, Pitch Yaw
  plt.subplot(8, 1, 3)
  plt.plot(sim_x_axis, sim[:plot_size,Roll], '-', label='sim Roll')
  plt.plot(sim_x_axis, sim[:plot_size,Pitch], '-', label='sim Pitch')
  plt.plot(sim_x_axis, sim[:plot_size,Yaw], '-', label='sim Yaw')
  plt.xlabel('Time [s]')
  plt.ylabel('Omega [rad/s]')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)


  plt.subplot(8, 1, 4)
  plt.plot(real_x_axis, np.radians(real[:plot_size,Roll]), '-', label='real Roll')
  plt.plot(real_x_axis, np.radians(real[:plot_size,Pitch]), '-', label='real Pitch')
  plt.plot(real_x_axis, np.radians(real[:plot_size,Yaw]), '-', label='real Yaw')
  plt.xlabel('Time [s]')
  plt.ylabel('Omega [rad/s]')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  # thrust 0
  plt.subplot(8, 1, 5)
  plt.plot(real_x_axis, real[:plot_size,t0], '-', label='real')
  plt.plot(sim_x_axis, sim[:plot_size,t0], '-', label='sim')
  plt.xlabel('Time [s]')
  plt.ylabel('thrust 0')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  # thrust 0
  plt.subplot(8, 1, 6)
  plt.plot(real_x_axis, real[:plot_size,t1], '-', label='real')
  plt.plot(sim_x_axis, sim[:plot_size,t1], '-', label='sim')
  plt.xlabel('Time [s]')
  plt.ylabel('thrust 1')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  # thrust 0
  plt.subplot(8, 1, 7)
  plt.plot(real_x_axis, real[:plot_size,t2], '-', label='real')
  plt.plot(sim_x_axis, sim[:plot_size,t2], '-', label='sim')
  plt.xlabel('Time [s]')
  plt.ylabel('thrust 2')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  # thrust 0
  plt.subplot(8, 1, 8)
  plt.plot(real_x_axis, real[:plot_size,t3], '-', label='real')
  plt.plot(sim_x_axis, sim[:plot_size,t3], '-', label='sim')
  plt.xlabel('Time [s]')
  plt.ylabel('thrust 3')
  plt.legend(loc=9, ncol=3, borderaxespad=0.)

  plt.show()




if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("sim_data", type=str, help="CSV file data")
  parser.add_argument("real_data", type=str, help="CSV file data")  
  args = parser.parse_args()

  dataType = float #np.dtype('Float64')

  sim = np.loadtxt(args.sim_data, delimiter=",", skiprows=1, dtype=dataType)
  real = np.loadtxt(args.real_data, delimiter=",", skiprows=1, dtype=dataType)

  main(sim, real)