import numpy as np
import matplotlib.pyplot as plt

from gym_art.quadrotor.sensor_noise import SensorNoise

noise = SensorNoise()
num_sample = 10000

## dt = 0.01
omega = np.zeros(3)
omega_100hz_noisy = [omega]
for i in range(num_sample):
	omega = noise.add_noise_to_omega(omega, 0.01)
	omega_100hz_noisy.append(omega)
	omega = np.zeros(3)

## dt = 0.005
omega = np.zeros(3)
omega_200hz_noisy = [omega]
for i in range(num_sample):
	omega = noise.add_noise_to_omega(omega, 0.005)
	# omega = noise.add_noise_to_omega(omega, 0.005)
	omega_200hz_noisy.append(omega)
	omega = np.zeros(3)


omega_100hz_noisy = np.array(omega_100hz_noisy)
omega_200hz_noisy = np.array(omega_200hz_noisy)

# new figure
plt.figure(0)

# Roll
plt.subplot(3, 1, 1)
plt.plot(omega_100hz_noisy[:, 0], '-', label='100 Hz')
plt.plot(omega_200hz_noisy[:, 0], '-', label='200 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Roll')
plt.legend(loc=9, ncol=3, borderaxespad=0.)

# Pitch
plt.subplot(3, 1, 2)
plt.plot(omega_100hz_noisy[:, 1], '-', label='100 Hz')
plt.plot(omega_200hz_noisy[:, 1], '-', label='200 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Pitch')
plt.legend(loc=9, ncol=3, borderaxespad=0.)

# Yaw
plt.subplot(3, 1, 3)
plt.plot(omega_100hz_noisy[:, 2], '-', label='100 Hz')
plt.plot(omega_200hz_noisy[:, 2], '-', label='200 Hz')
plt.xlabel('Time [s]')
plt.ylabel('Yaw')
plt.legend(loc=9, ncol=3, borderaxespad=0.)

plt.show()
