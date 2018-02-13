import sys
import math

import numpy as np

sys.path.append("./cutils/")

import cutil



def L1partition_true(x, epsilon, ratio=0.5, gethist=False):
	n = len(x)
	hist = cutil.L1partition_true(n, x, epsilon, ratio, np.random.randint(500000))
	hist = np.append(hist, 0)
	hatx = np.zeros(n)
	rb = n
	bucks = []

	for lb in hist[1:]:
		bucks.insert(0, [lb, rb-1])
		rb = lb
		if lb == 0:
			break
	
	return bucks


def L1partition(x, epsilon, ratio=0.5, gethist=False):
	"""Compute the noisy L1 histogram using all interval buckets
	
	Args:
		x - list of numeric values. The input data vector
		epsilon - double. Total private budget
		ratio - double in (0, 1). use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition
		gethist - boolean. If set to truth, return the partition directly (the privacy budget used is still ratio*epsilon)
		
	Return:
		if gethist == False, return an estimated data vector. Otherwise, return the partition
	"""
	n = len(x)
	hist = cutil.L1partition(n, x, epsilon, ratio, np.random.randint(500000))
	hist = np.append(hist, 0)
	hatx = np.zeros(n)
	rb = n
	print( hist)
	if gethist == True:
		bucks = []
		for lb in hist[1:]:
			bucks.insert(0, [lb, rb-1])
			rb = lb
			if lb == 0:
				break
		return bucks
	else:
		for lb in hist[1:]:
			if (epsilon == np.inf):
				hatx[lb:rb] = max(0, sum(x[lb:rb]) ) / float(rb - lb)
			else:
				hatx[lb:rb] = max(0, sum(x[lb:rb]) + np.random.laplace(0, 1.0/(epsilon*(1-ratio)), 1)) / float(rb - lb)	
			
			rb = lb
			if lb == 0:
				break
	
		return hatx

def L1partition_approx(x, epsilon, ratio=0.5, gethist=False):
	"""Compute the noisy L1 histogram using interval buckets of size 2^k
	
	Args:
		x - list of numeric values. The input data vector
		epsilon - double. Total private budget
		ratio - double in (0, 1) the use ratio*epsilon for partition computation and (1-ratio)*epsilon for querying
				the count in each partition
		gethist - boolean. If set to truth, return the partition directly (the privacy budget used is still ratio*epsilon)
		
	Return:
		if gethist == False, return an estimated data vector. Otherwise, return the partition
	"""
	n = len(x)
	#print n
	hist = cutil.L1partition_approx(n, x, epsilon, ratio, np.random.randint(500000))

#	if hist[-1] != 0:
#		print 'l1partition.py: Added an extra zero'
#		hist = np.insert(hist, len(hist), 0)

	hatx = np.zeros(n)
	rb = n
	if gethist:
		bucks = []
		for lb in hist[1:]:
			bucks.insert(0, [lb, rb-1])
			rb = lb
			if lb == 0:
				break
		print( bucks)
		return bucks
	else:
		for lb in hist[1:]:
			if (epsilon == np.inf):
				hatx[lb:rb] = max(0, sum(x[lb:rb]) ) / float(rb - lb)
			else:
				hatx[lb:rb] = max(0, sum(x[lb:rb]) + np.random.laplace(0, 1.0/(epsilon*(1-ratio)), 1)) / float(rb - lb)	
			rb = lb
			if lb == 0:
				break

		return hatx
