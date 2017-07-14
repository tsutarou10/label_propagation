import numpy as np
import os
from sklearn.metrics import recall_score,precision_score,f1_score
from pulp import *
from itertools import product
import random

class Lnp:
	def __init__(self):
		self.K = 10
		self.a = 0.99

	def set_features_labels(self):
		labels = np.load("./all_labels.npy")
		features = np.load("all_features.npy")

		return features,labels

	def lnp(self):

		scores1 = []
		scores2 = []
		scores3 = []

		for k in range(10):
			X,Y = self.set_features_labels()
			Y = np.array(Y,dtype = float)
			V = Y.shape[0]
			I = np.matrix(np.identity(V))			
			indices = np.random.permutation(X.shape[0])
			X = X[indices]
			Y = Y[indices]
			Xl,Xu,Yl,Yu = train_test_split(X,Y,test_size = 0.95,random_state = 1)
			Yu_ = np.zeros((Xu.shape[0],self.cat_num))
			X = np.r_[Xl,Xu]

			W = np.zeros([V,V])

			dis_array = np.zeros([V,V],dtype = float)
			H = np.tile(np.diag(np.dot(X,X.T)),(V,1))
			G = np.dot(X,X.T)
			dis_array = (H - 2 * G + H.T)
			nearidx = np.argsort(dis_array,axis = 1)
			nearidx2 = np.zeros([V,self.K+1])

			for i in range(V):
				for j in range(self.K+1):
					nearidx2[i][j] = nearidx[i][V-j-1]

			gram_matrix = np.zeros([V,V],dtype = float)

			for i in range(V):
				for k in range(self.K):
					gram_matrix[i][int(nearidx2[i][k+1])] = np.dot(X[i] - X[int(nearidx2[i][k+1])],X[i] - X[int(nearidx2[i][k+1])])
			wi = V
			wj = self.K
			pr = product(range(wi),range(wj))
			w = np.array([LpVariable("w%d%d"%(i,j),lowBound = 0.0,upBound = 1.0,cat=LpContinuous) for i,j in pr])
			w = w.reshape(wi,wj)
			W = np.zeros([V,V],dtype = float)
			for i in range(V):
				obj_func = LpProblem()
				obj_func += lpSum((w[i][j] * gram_matrix[i][int(nearidx2[i][j+1])] for j in range(wj)))
				obj_func += lpSum(w[i][j] for j in range(wj)) == 1.0
				
				for j in range(wj):
					obj_func += w[i][j] >= 0.0
				
				obj_func.solve()
				for j in range(wj):
					W[i][int(nearidx2[i][j+1])] = value(w[i][j])

			P = W
			Y_prev = np.r_[Yl,Yu]
			f0 = Y_prev
			Y_prevprev = Y_prev
			Y_next = []
			t = 0
			for t in range(1000):
				Y_next = self.a * np.dot(P,Y_prev) + (1 - self.a) * f0
				Y_prev = Y_next
				t += 1
			Y_next_max = np.max(Y_next,axis=1)[:,np.newaxis]
			Y_next[np.where(Y_next >= Y_next_max * i * 0.01)] = 1
			Y_next[np.where(Y_next < Y_next_max * i * 0.01)] = 0
			score1 = precision_score(Y_next[int(Y.shape[0] * 0.05):],Yu,average = "micro")
			score2 = recall_score(Y_next[int(Y.shape[0] * 0.05):],Yu,average = "micro")
			score3 = f1_score(Y_next[int(Y.shape[0] * 0.05):],Yu,average = "micro")
			scores1.append(score1)
			scores2.append(score2)
			scores3.append(score3)

		print "micro precision    : %.2f" % np.array(scores1).mean()
		print "micro recall    : %.2f" % np.array(scores2).mean()
		print "micro F1        : %.2f" % np.array(scores3).mean()


if __name__ == "__main__":
	l = Lnp()
	l.lnp()