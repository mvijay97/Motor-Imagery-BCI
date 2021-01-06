import scipy.io
import scipy
import numpy as np
import mne
from sklearn.model_selection import train_test_split


channels=22
bands=10

def segement_signal(data,labels,position):
	data_temp=[]
	labels_temp=[]

	for i in range(len(labels)):
		if labels[i] in [769,770,771,772] and labels[i-1]!=1023:
			labels_temp.append(labels[i])
			start=position[i]+250
			end=start+750
			data_temp.append(data[start:end])

	data=[]
	for i in data_temp:
		data.append(np.transpose(i))
	data=np.array(data)

	return(data,labels_temp)

def impute_nan(data,n):
	for j in range(data.shape[0]):
		for i in range(channels):
			for k in range(750):
				if np.isnan(data[j][i][k]):
					mean=sum(data[j][i][k-1:k-n-1:-1])/n
					data[j][i][k]=mean
	return(data)

def filter_signal(data):
	filter_bank=[(0,4),(4,8),(8,12),(12,16),(16,20),(20,24),(24,28),(28,32),(32,36),(36,40)]
	data_fil=[]
	for i in filter_bank:
		temp=[]
		for j in data:
			fil=mne.filter.filter_data(j,250,i[0],i[1])
			temp.append(fil)
		data_fil.append(temp)
	data_fil=np.array(data_fil)
	return(data_fil)


for subject in range(1,10):
	print(subject)
	path="data/A0{}T.mat".format(subject)
	mat=scipy.io.loadmat(path)

	data=mat['s']
	data=np.delete(data,[22,23,24],1) #drop columns corresponding to EOG

	labels=mat['t']
	labels=[mat['t'][i][0] for i in range(mat['t'].shape[0])]

	position=mat['p']
	position=[mat['p'][i][0] for i in range(mat['p'].shape[0])]

	data,labels=segement_signal(data,labels,position)
	data=impute_nan(data,5)

	X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)
	
	X_train=filter_signal(X_train)
	X_test=filter_signal(X_test)

	np.save("filtered_data/A0{}T_X_train".format(subject),X_train)
	np.save("filtered_data/A0{}T_y_train".format(subject),y_train)
	np.save("filtered_data/A0{}T_X_test".format(subject),X_test)
	np.save("filtered_data/A0{}T_y_test".format(subject),y_test)