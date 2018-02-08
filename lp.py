# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from sklearn import datasets
import os
from sklearn.metrics import recall_score,precision_score,f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class LabelPropagation:

	def __init__(self):
		'''
		set hyper parametars
		'''
		self.m = 1.0

	def lp(self):
		'''
		Label Propagation
		'''

		trY = np.load('training_labels.npy') #Label of training data
		teY = np.load('test_labels.npy') # Label of test data 
		Y = np.r_[trY,teY]
		V = Y.shape[0] # the number of data
		I = np.matrix(np.identity(V))
		P = np.load('P.npy') # load the probabilistic transition matrix
		Yl = Y[0:int(trY.shape[0])] # labeled data
		Yu = np.zeros([V-int(trY.shape[0]),Y.shape[1]]) #unlabeled data

		iterNum = 0 # the number of current iteration

		Y_prev = np.r_[Yl,Yu]
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

		trX = np.load('training_features.npy') #the features of training data
		teX = np.load('test_features.npy')	#the features of test data

		X = np.r_[trX,teX]

		V = X.shape[0] #the number of data

		P = np.zeros([V,V]) #probabilistic transition matrix
		W = np.zeros([V,V]) #weight matrix
		
		dis_array = np.zeros([V,V],dtype = float)
		H = np.tile(np.diag(np.dot(X,X.T)),(V,1))
		G = np.dot(X,X.T)
		dis_array = H - 2 * G + H.T

		W = np.exp(-1 * dis_array / self.m)
		for i in range(W.shape[0]):
			W[i][i] = 0.0
		W_sum = np.sum(W,axis = 1)

		P = W / W_sum[:,np.newaxis]
		
		np.save("P.npy",P) #save posibility transition matrix
		np.save("W.npy",W) #save weight
		print 'finish making parametars!'

	def set_par(self):
		'''
		set parametars for label propagation
		'''
		P = np.load("P.npy")

		return P

	def metrics_lp(self):
		'''
		'''
		thresholds = 0.4 #thresholds for label assignment after the iteration of LP
		
		trY = np.load('training_labels.npy') # label of training data
		teY = np.load('test_labels.npy') # label of test data
		y_true = np.r_[trY,teY]

		y_pred = self.lp() # assign a label
		y_pred[np.where(y_pred > thresholds)] = 1
		y_pred[np.where(y_pred <= thresholds)] = 0
		microrecall = recall_score(teY, y_pred[int(trY.shape[0]):], average='micro')
		microprecision = precision_score(teY, y_pred[int(trY.shape[0]):], average='micro')
		microf1 = f1_score(teY, y_pred[int(trY.shape[0]):], average='micro')
			
		print 'micro recall score : ' + str(microrecall)
		print 'micro precision score : ' + str(microprecision)
		print 'micro f1 score : ' + str(microf1)

if __name__ == "__main__":
	lp = LabelPropagation()
	lp.make_par() # make parametars for label propagation
	lp.metrics_lp() # print micro recall, precision and f1 score of label propagation 