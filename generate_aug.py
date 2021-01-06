import numpy as np
import scipy.io as spi
import scipy
import os
import random
from scipy.interpolate import Rbf
import time
import matplotlib.pyplot as plt

base_path="./data/filtered/"


data_path = sorted(os.listdir("./data/filtered/signals"))
label_path= sorted(os.listdir("./data/filtered/labels"))

data_path = ['X01T.npy',  'X07T.npy', 'X08T.npy']
label_path = ['y01T.npy', 'y07T.npy', 'y08T.npy']

print(data_path)
print(label_path)
data_save_path = "./data/filtered/aug/signals/"
label_save_path= "./data/filtered/aug/labels/"


gaussian = False
cyclic_shift_f = False
cyclic_shift_b = False
dc_shift = False
rotation_x = True
rotation_y = True
rotation_z = True
translation = True


#Gaussian Parameters
varience_1 = [0.001, 0.01, 0.02, 0.03]
varience_2 = [0.1, 0.2, 0.3, 0.5]
mean = 0

#Cyclic Shift Parameters
#Forward Shift and Backward Shift
cyclic_forward = [-30,-60,-90,-120] ## [1,2,3,4,5,6] shift by 2 will be [6,5,1,2,3,4]
cyclic_backward = [30,60,90,120]

#DC Shift?
dc_shift_arr = [0.1,0.3,0.5,-0.1,-0.3,-0.5]

#rotation parameters

electrode_map_x = [0,-7,-3.5,0,3.5,7,-10.5,-7,-3.5,0,3.5,7,10.5,-7,-3.5,0,3.5,7,-3.5,0,3.5,0] 
electrode_map_y = [7,3.5,3.5,3.5,3.5,3.5,0,0,0,0,0,0,0,-3.5,-3.5,-3.5,-3.5,-3.5,-7,-7,-7,10.5]
# electrode_map_z = [-1.796, -1.796, -0.898, -0.898, -0.898, -1.796, -2.694, -1.796, -0.898, 0, -0.898, -1.796, -2.694, -1.796, -0.898, -0.898, -0.898, -1.796, -1.796, -1.796, -1.796, -2.694]
electrode_map_z = [-2.6091, -2.6091, 1.3046, 1.3046, 1.3046, -2.6091, -3.913, -2.6091, 1.3046, 0, 1.3046, -2.6091, -3.913, -2.6091, 1.3046, 1.3046, 1.3046, -2.6091, -2.6091, -2.6091, -2.6091, -3.913]

xyz= np.array([electrode_map_x,electrode_map_y,electrode_map_z])


#rotation_parameters
X_rot_angles = [25]
Y_rot_angles = [10]
Z_rot_angles = [10]
translation_varience = [0.1, 0.3, 0.4, 0.03, 0.05]




for subject in range(len(data_path)):
	signal = np.load("./data/filtered/signals/" + data_path[subject])
	labels = np.load("./data/filtered/labels/"+ label_path[subject])
	print(data_path[subject])
	print(label_path[subject])
	augmented_signal_gaussian_1 = np.zeros((10,288,22,750))
	augmented_signal_gaussian_2 = np.zeros((10,288,22,750))
	augmented_signal_cyclic_f   = np.zeros((10,288,22,750))
	augmented_signal_cyclic_b   = np.zeros((10,288,22,750))
	augmented_signal_dc_shift   = np.zeros((10,288,22,750))


	for band in range(10):
		for trial in range(288):
			for channel in range(22):
				# print(band,trial,channel)
				if gaussian:
					varience_rv = random.randint(0,len(varience_1)-1)
					augmented_signal_gaussian_1[band,trial,channel,:] = signal[band,trial,channel,:] + np.random.normal(0,varience_1[varience_rv],750)
					augmented_signal_gaussian_2[band,trial,channel,:] = signal[band,trial,channel,:] + np.random.normal(0,varience_2[varience_rv],750)

				if cyclic_shift_f:
					cyclic_shift_f_rv = random.randint(0,len(cyclic_forward)-1)
					augmented_signal_cyclic_f[band,trial,channel,:] = np.roll(signal[band,trial,channel,:],cyclic_forward[cyclic_shift_f_rv])

				if cyclic_shift_b:
					cyclic_shift_b_rv = random.randint(0,len(cyclic_backward)-1)
					augmented_signal_cyclic_b[band,trial,channel,:] = np.roll(signal[band,trial,channel,:],cyclic_backward[cyclic_shift_b_rv])

				if dc_shift:
					dc_shift_rv = random.randint(0,len(dc_shift_arr)-1)
					augmented_signal_dc_shift[band,trial,channel,:] = signal[band,trial,channel,:] + dc_shift_arr[dc_shift_rv]




	if gaussian:
		augmented_signal_gaussian = np.concatenate((signal,augmented_signal_gaussian_1,augmented_signal_gaussian_2), axis = 1)
		augmented_label = np.concatenate((labels,labels,labels))
		np.save(data_save_path + "gaussian/" + data_path[subject],augmented_signal_gaussian)
		np.save(label_save_path + "gaussian/" + label_path[subject],augmented_label)

	if cyclic_shift_f:
		augmented_signal_cyclic_f = np.concatenate((signal,augmented_signal_cyclic_f), axis =1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "cyclic_shift_f/"+ data_path[subject], augmented_signal_cyclic_f)
		np.save(label_save_path + "cyclic_shift_f/" + label_path[subject],augmented_label)

	if cyclic_shift_b:
		augmented_signal_cyclic_b = np.concatenate((signal,augmented_signal_cyclic_b), axis =1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "cyclic_shift_b/"+ data_path[subject], augmented_signal_cyclic_b)
		np.save(label_save_path + "cyclic_shift_b/" + label_path[subject],augmented_label)

	if dc_shift:
		augmented_signal_dc_shift = np.concatenate((signal,augmented_signal_dc_shift), axis =1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "dc_shift/"+ data_path[subject], augmented_signal_dc_shift)
		np.save(label_save_path + "dc_shift/" + label_path[subject],augmented_label)

print("starting rotation augmentation")

for subject in range(len(data_path)):
	signal = np.load("./data/filtered/signals/" + data_path[subject])
	labels = np.load("./data/filtered/labels/"+ label_path[subject])
	print(data_path[subject])
	print(label_path[subject])

	augmented_rotation_x = np.zeros((10,288,22,750))
	augmented_rotation_y = np.zeros((10,288,22,750))
	augmented_rotation_z = np.zeros((10,288,22,750))
	augmented_translation = np.zeros((10,288,22,750))


	start=time.time()
	for band in range(10):
		print(band)
		for trial in range(288):
			for time_stamp in range(750):
				rbfi = Rbf(electrode_map_x, electrode_map_y,electrode_map_z, signal[band,trial,:,time_stamp], function= "gaussian")

				if rotation_x:
					rotation_x_rv = random.randint(0,len(X_rot_angles)-1)
					angle_rad = X_rot_angles[rotation_x_rv] * np.pi / 180
					R = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])
					yz = np.array([electrode_map_y,electrode_map_z])
					new_yz = np.matmul(R,yz)
					aug_temp = rbfi(electrode_map_x, new_yz[0,:], new_yz[1,:])
					augmented_rotation_x[band,trial,:,time_stamp] = aug_temp


				if rotation_y:
					rotation_y_rv = random.randint(0,len(Y_rot_angles)-1)
					angle_rad = Y_rot_angles[rotation_y_rv] * np.pi / 180
					R = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])
					xz = np.array([electrode_map_x,electrode_map_z])
					new_xz = np.matmul(R,xz)
					aug_temp = rbfi(new_xz[0,:],electrode_map_y, new_xz[1,:])
					augmented_rotation_y[band,trial,:,time_stamp] = aug_temp

				if rotation_z:
					rotation_z_rv = random.randint(0,len(Z_rot_angles)-1)
					angle_rad = Z_rot_angles[rotation_y_rv] * np.pi / 180
					R = np.array([[np.cos(angle_rad),-np.sin(angle_rad)],[np.sin(angle_rad),np.cos(angle_rad)]])
					xy = np.array([electrode_map_x,electrode_map_y])
					new_xy = np.matmul(R,xy)
					aug_temp = rbfi(new_xy[0,:], new_xy[1,:], electrode_map_z)
					augmented_rotation_z[band,trial,:,time_stamp] = aug_temp

				if translation:
					translation_rv = random.randint(0,len(translation_varience)-1)
					new_xyz = xyz + np.random.normal(0,translation_varience[translation_rv],[3,22])
					aug_temp = rbfi(new_xyz[0,:],new_xyz[1,:], new_xyz[2,:])
					augmented_translation[band,trial,:,time_stamp] = aug_temp




	if rotation_x:
		augmented_rotation_x = np.concatenate((signal,augmented_rotation_x), axis =1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "rotation_x/"+ data_path[subject], augmented_rotation_x)
		np.save(label_save_path + "rotation_x/" + label_path[subject],augmented_label)
	if rotation_y:
		augmented_rotation_y = np.concatenate((signal,augmented_rotation_y), axis =1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "rotation_y/"+ data_path[subject], augmented_rotation_y)
		np.save(label_save_path + "rotation_y/" + label_path[subject],augmented_label)
	if rotation_z:
		augmented_rotation_z = np.concatenate((signal,augmented_rotation_z), axis =1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "rotation_z/"+ data_path[subject], augmented_rotation_z)
		np.save(label_save_path + "rotation_z/" + label_path[subject],augmented_label)
	if translation:
		augmented_translation = np.concatenate((signal, augmented_translation), axis=1)
		augmented_label = np.concatenate((labels,labels))
		np.save(data_save_path + "translation/"+ data_path[subject], augmented_translation)
		np.save(label_save_path + "translation/" + label_path[subject],augmented_label)


end=time.time()
print(end-start)




