# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from sklearn import datasets
import os
from sklearn.metrics import recall_score,precision_score,f1_score
from tqdm import tqdm
import matplotlib.pyplot as plt

class DynamicLabelPropagation:

	def __init__(self):
		'''
		set hyper parametars
		'''
		self.m = 1.0
		self.K = 10
		self.a = 0.05
		self.l = 0.1
		self.epsiron = 1e-11

		self.trainingFeatures = 'training_features.npy'
		self.testFeatures = 'test_features.npy'
		self.trainingLabels = 'training_labels.npy'
		self.testLabels = 'test_labels.npy'

	def dlp(self):
		'''
		Dynamic Label Propagation
		'''
		trY, teY = self.set_labels() # training and test labels
		Y = np.r_[trY,teY]
		V = Y.shape[0] # the number of data
		I = np.matrix(np.identity(V))
		P, PP = self.set_par() # load the probabilistic transition matrix and fusion matrix
		Yl = Y[0:int(trY.shape[0])] # labeled data
		Yu = np.zeros([V-int(trY.shape[0]),Y.shape[1]]) #unlabeled data

		iterNum = 0 # the number of current iteration

		Y_prev = np.r_[Yl,Yu]
		Y_next = []

		while 1:
			Y_next = P.dot(Y_prev)
			Y_next[0:int(trY.shape[0])] = Yl #clamping
			print 'iter:' + str(iterNum) + '->' + 'error rate : ' + str((abs(Y_prev - Y_next)).sum())

			if (abs(Y_prev - Y_next)).sum() < self.epsiron: #convergence
				break

			P = (PP.dot(P + self.a * Y_prev.dot(Y_prev.T))).dot(PP.T) + self.l * I
			
			for i in range(P.shape[0]):
				P[i] = self.softmax(P[i])

			Y_prev = Y_next
			iterNum += 1

		return Y_next

	def set_par(self) :
		'''
		set parametars for label propagation
		'''

		P = np.load('./parameters/P.npy')
		PP = np.load('./parameters/PP.npy')

		return P, PP

	def softmax(self,x):
		'''
		softmax function
		'''
		return x / (np.sum(x) + 1e-11)

	def make_par(self):
		'''
		make parameters for dynamic label propagation
		'''
		
		trX, teX = self.set_features() # training and test features
		X = np.r_[trX,teX]

		V = X.shape[0] #the number of data

		P = np.zeros([V,V]) #probabilistic transition matrix
		W = np.zeros([V,V]) #weight matrix
		PP = np.zeros([V,V])
		WW = np.zeros([V,V])

		dis_array = np.zeros([V,V],dtype = float)
		H = np.tile(np.diag(np.dot(X,X.T)),(V,1))
		G = np.dot(X,X.T)
		dis_array = H - 2 * G + H.T

		W = np.exp(-1 * dis_array / self.m)
		print 'making W!'

		for i in range(W.shape[0]):
			W[i][i] = 0.0
		W_sum = np.sum(W,axis = 1)

		P = W / W_sum[:,np.newaxis]
		print 'making P!'

		nearidx = np.argsort(dis_array,axis=1)
		
		for i in range(V):
			for k in range(self.K):
				WW[i][nearidx[i][k+1]] = W[i][nearidx[i][k+1]]
		print 'making WW!'

		WW_sum = np.sum(WW,axis=1)
		PP = WW / WW_sum[:,np.newaxis]
		print 'making PP!'

		if(not os.path.isdir('./parameters')) :
			os.mkdir('./parameters')

		np.save('./parameters/P.npy', P) #save posibility transition matrix
		np.save('./parameters/W.npy', W) #save weight
		np.save('./parameters/WW.npy', WW)
		np.save('./parameters/PP.npy', PP)

		print 'finish making parametars!'

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
		print each static score
		'''
		thresholds = 0.5 #thresholds for dynamic label assignment after the iteration of LP
		
		trY, teY = self.set_labels() # label of training and test data

		y_pred = self.dlp() # assign a label
		y_pred[np.where(y_pred > thresholds)] = 1
		y_pred[np.where(y_pred <= thresholds)] = 0
		microrecall = recall_score(teY, y_pred[int(trY.shape[0]):], average='micro')
		microprecision = precision_score(teY, y_pred[int(trY.shape[0]):], average='micro')
		microf1 = f1_score(teY, y_pred[int(trY.shape[0]):], average='micro')
			
		print 'micro recall score : ' + str(microrecall)
		print 'micro precision score : ' + str(microprecision)
		print 'micro f1 score : ' + str(microf1)

if __name__ == '__main__':
	dlp = DynamicLabelPropagation()
	dlp.make_par() # make parametars for dynamic label propagation
	dlp.metrics_lp() # print micro recall, precision and f1 score of label propagation 
