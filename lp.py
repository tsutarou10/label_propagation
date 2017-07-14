# -*- coding: utf-8 -*-
#!/usr/bin/python

import numpy as np
from sklearn.metrics import recall_score,precision_score,f1_score
from sklearn.model_selection import train_test_split

class Lp:
	def __init__(self):
		self.K = 30
		self.T = 50
		self.l = 0.1
		self.m = 1.0
		self.cat_num = 22

	def set_features_labels(self):
		
		labels = np.load("all_labels.npy")
		features = np.load("all_features.npy")
		return features,labels

	def lp(self):
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
			self.make_par(X)
			P,W,PP,WW = self.set_par()
			
			Y_prev = np.r_[Yl,Yu_]
			Y_next = []

			scores1 = []
			scores2 = []
			scores3 = []

			for t in range(self.T):
				Y_next = P.dot(Y_prev)
				Y_next[0:int(Y.shape[0] * 0.05)] = Yl
				Y_prev = Y_next

			i = 80
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

	def make_par(self,X):
		V = X.shape[0]
		P = np.zeros([V,V])
		PP = np.zeros([V,V])
		W = np.zeros([V,V])
		WW = np.zeros([V,V])

		dis_array = np.zeros([V,V],dtype = float)
		H = np.tile(np.diag(np.dot(X,X.T)),(V,1))
		G = np.dot(X,X.T)
		dis_array = H - 2 * G + H.T

		W = np.exp(-1 * dis_array ** 2 / self.m)
		W_sum = np.sum(W,axis = 1)

		print "make W"

		P = W / W_sum[:,np.newaxis]
		print "make P"

		np.save("P.npy",P)
		np.save("W.npy",W)

	def set_par(self):
		P = np.load("P.npy")
		W = np.load("W.npy")

		return P,W

if __name__ == "__main__":
	l = Lp()
	l.Lp()
	print "finish"