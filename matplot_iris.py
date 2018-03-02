#coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def main():
	features = np.load('test_features.npy')
	labels = np.load('y_pred_of_lp.npy')

	pca = PCA(n_components = 2)

	features = pca.fit_transform(features)

	color = ['red','blue','green']
	for number,l in enumerate(labels):
		xaxis = features[number][0]
		yaxis = features[number][1]
		idx = np.argmax(l)
		plt.scatter(xaxis,yaxis,color = color[idx])
	plt.show()

if __name__ == '__main__':
	main()