# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from sklearn import datasets
import os
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class LabelPropagation:

	def __init__(self):
		'''
		set hyper parametars
		'''
		self.m = 1.0

		self.trainingFeatures = 'training_features.npy'
		self.testFeatures = 'test_features.npy'
		self.trainingLabels = 'training_labels.npy'
		self.testLabels = 'test_labels.npy'

	def lp(self):
		'''
		Label Propagation
		'''

		trY, teY = self.set_labels() # trainig and test labels
		Y = np.r_[trY, teY]
		V = Y.shape[0] # the number of data
		P = self.set_par() # load the probabilistic transition matrix and matrix

		Yl = Y[0:int(trY.shape[0])] # labeled data
		Yu = np.zeros([V-int(trY.shape[0]), Y.shape[1]]) #unlabeled data

		iterNum = 0 # the number of current iteration

		Y_prev = np.r_[Yl, Yu]
		Y_next = []

		while 1:
			Y_next = P.dot(Y_prev)
			
			Y_next[0:int(trY.shape[0])] = Yl #clamping
			print 'iter:' + str(iterNum) + '->' + 'error rate : ' + str((abs(Y_prev - Y_next)).sum())

			if (Y_next == Y_prev).all(): #convergence
				break

			Y_prev = Y_next
			iterNum += 1

		return Y_next

	def make_par(self):
		'''
		make parameters for label propagation
		'''

		trX, teX = self.set_features() # training and test features
		X = np.r_[trX, teX]

		V = X.shape[0] #the number of data

		P = np.zeros([V, V]) #probabilistic transition matrix
		W = np.zeros([V, V]) #weight matrix
		
		dis_array = np.zeros([V, V], dtype = float)
		H = np.tile(np.diag(np.dot(X, X.T)), (V, 1))
		G = np.dot(X, X.T)
		dis_array = H - 2 * G + H.T

		W = np.exp(-1 * dis_array / self.m)
		print 'making W!'

		for i in range(W.shape[0]):
			W[i][i] = 0.0
		W_sum = np.sum(W, axis = 1)

		P = W / W_sum[:, np.newaxis]
		print 'making P!'

		if(not os.path.isdir('./parameters')) :
			os.mkdir('./parameters')

		np.save('./parameters/P.npy', P) #save posibility transition matrix
		np.save('./parameters/W.npy', W) #save weight
		print 'finish making parametars!'

	def set_par(self):
		'''
		set parametars for label propagation
		'''
		P = np.load('./parameters/P.npy')

		return P

	def set_features(self) :
		trX = np.load(self.trainingFeatures)
		teX = np.load(self.testFeatures)

		return trX, teX

	def set_labels(self) :
		trY = np.load(self.trainingLabels)
		teY = np.load(self.testLabels)

		return trY, teY

	def metrics_lp(self):
		'''
		'''
		#thresholds = 0.4 #thresholds for label assignment after the iteration of LP
		
		trY, teY = self.set_labels() # label of training and test data
		y_true = np.r_[trY, teY]

		y_pred = self.lp() # assign a label
		
		np.save('y_pred_of_lp.npy', y_pred[int(trY.shape[0]):])

		y_pred_max = np.argmax(y_pred, axis = 1)[:, np.newaxis]
		y_pred_max = y_pred_max.reshape(y_pred_max.shape[0], y_pred_max.shape[1])
		
		for number, ypm in enumerate(y_pred_max):
			if ypm == 0:
				n = [1, 2]
			elif ypm == 1:
				n = [0, 2]
			else:
				n = [0, 1]
			y_pred[number][ypm] = 1
			y_pred[number][n] = 0

		print y_pred
		microrecall = recall_score(teY,  y_pred[int(trY.shape[0]):], average='micro')
		microprecision = precision_score(teY,  y_pred[int(trY.shape[0]):], average='micro')
		microf1 = f1_score(teY,  y_pred[int(trY.shape[0]):], average='micro')
			
		print 'micro recall score : ' + str(microrecall)
		print 'micro precision score : ' + str(microprecision)
		print 'micro f1 score : ' + str(microf1)

if __name__ == '__main__':
	lp = LabelPropagation()
	lp.make_par() # make parametars for label propagation
	lp.metrics_lp() # print micro recall,  precision and f1 score of label propagation 
