import numpy as np
import os
from sklearn.metrics import recall_score,precision_score,f1_score
from pulp import *
from itertools import product
from sklearn import datasets
import random
from scipy.optimize import minimize
from cvxopt import matrix
import cvxopt
import gc
from tqdm import tqdm
class Lnp:
	def __init__(self):
		self.K = 10
		self.a = 1.0
		self.epsiron = 1e-11
		
	def make_weight(self):
		
		trX = np.load('training_features.npy')
		teX = np.load('test_features.npy')
		trY = np.load('training_labels.npy')
		teY = np.load('test_labels.npy')
		X = np.r_[trX,teX]
		V = X.shape[0]

		W = np.zeros([V,V])

		dis_array = np.zeros([V,V],dtype = float)
		H = np.tile(np.diag(np.dot(X,X.T)),(V,1))
		G = np.dot(X,X.T)
		dis_array = (H - 2 * G + H.T)
		nearidx = np.argsort(dis_array,axis = 1)

		
		
		N = V / 2
		div = V / N
		pbar = tqdm(total = div)
		for n in range(div):
			gram_matrix2 = np.zeros((self.K*N,self.K*N),dtype=float)
			for i in range(N):
				for j in range(self.K):
					for k in range(self.K):
						gram = np.dot(X[i]-X[nearidx[i][j+1]],X[i]-X[nearidx[i][k+1]])
						gram_matrix2[i*self.K + j][i*self.K + k] = gram
			P = matrix(gram_matrix2)
			q = matrix(np.zeros(self.K*N,dtype = float))
			G = np.zeros((N,self.K*N),dtype=float)
			for a in range(N):
				for b in range(self.K):
					G[a][self.K*a+b] = 1.
			A = matrix(G)
			b = matrix(np.ones(N,dtype = float))
			G = np.matrix(np.identity(self.K*N))
			G = matrix(-G)
			h = matrix(np.zeros(N*self.K,dtype=float))
			solve = cvxopt.solvers.qp(P,q,G,h,A,b)
			for i in range(N):
				for j in range(self.K):
					W[n*N+i][nearidx[i][j+1]] = solve['x'][i*self.K+k]
			
			pbar.update(1)

			number = 0
			for j in range(V):
				if j in nearidx[i][1:self.K]:
					W[i][j] = solve['x'][number]
					number += 1


			pbar.update(1)
		pbar.close()
		np.save('lnpW.npy',W)

	def lnp(self):
		W = np.load('lnpW.npy')
		trY = np.load('training_labels.npy')
		teY = np.load('test_labels.npy')
		Y = np.r_[trY,teY]

		V = Y.shape[0]

		Yl = Y[0:int(trY.shape[0])] # labeled data
		Yu = np.zeros([V-int(trY.shape[0]),Y.shape[1]]) #unlabeled data
		
		P = W
		Y_prev = np.r_[Yl,Yu]
		f0 = Y_prev
		Y_next = []

		iterNum = 0
		while 1:
			Y_next = self.a * np.dot(P,Y_prev) + (1 - self.a) * f0
			print 'iter:' + str(iterNum) + '->' + 'error rate : ' + str((abs(Y_prev - Y_next)).sum())
			
			if (abs(Y_prev - Y_next)).sum() < self.epsiron: #convergence
				break
			Y_prev = Y_next
			iterNum += 1

		return Y_next

	def softmax(self,x):
		return x / (np.sum(x))

	def metrics_lp(self):
		
		trY = np.load('training_labels.npy')
		teY = np.load('test_labels.npy')
		y_pred = self.lnp()

		thresholds = 0.6
		for i in range(y_pred.shape[0]):
			y_pred[i] = self.softmax(y_pred[i])
		print y_pred
		y_pred[np.where(y_pred > thresholds)] = 1
		y_pred[np.where(y_pred <= thresholds)] = 0
		microrecall = recall_score(teY, y_pred[int(trY.shape[0]):], average='micro')
		microprecision = precision_score(teY, y_pred[int(trY.shape[0]):], average='micro')
		microf1 = f1_score(teY, y_pred[int(trY.shape[0]):], average='micro')
			
		print 'micro recall score : ' + str(microrecall)
		print 'micro precision score : ' + str(microprecision)
		print 'micro f1 score : ' + str(microf1)

if __name__ == "__main__":
	l = Lnp()
	#l.make_weight()
	l.metrics_lp()


