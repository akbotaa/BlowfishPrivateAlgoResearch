"""Sample code for using the Data- and Workload-Aware (DAWA) algorithm and its sub-algorithms."""

import numpy

import dawa
import l1partition

def main():
	print """In this sample, we use the following data vector x
	x = [2, 3, 8, 1, 0, 2, 0, 4, 2, 4]
	"""
	x = [2, 3, 8, 1, 0, 2, 0, 4, 2, 4]
	
	print """Compute a data-aware histogram of x using the L1 parition algorithm, with the following parameters:
	epsilon: 1
	amount of privacy budget used to compute the partition: 25%"""
	# Use the L1 parition algorithm to divide x into buckets
	# Fix the random seeds to get the consistent results in two runs
	numpy.random.seed(10)
	
	print "The buckets in the histogram are:"
	# Run the L1 partition algorithm but only ask for the buckets
	print "\t", l1partition.L1partition(x, 1, 0.25, gethist=True)

	numpy.random.seed(10)
	print "The estimated data vector is:"
	# Run the L1 partition algorithm again, and ask for the estimated data vector
	print "\t", l1partition.L1partition(x, 1, 0.25)
	print
	
	print "Compute a data-aware histogram of x using the approximated L1 partition algorithm \
	(the first step of DAWA), with same parameters."
	
	# Use the approximate L1 parition algorithm to divide x into buckets
	# Fix the random seeds to get the consistent results in two runs
	numpy.random.seed(10)
	
	print "The buckets in the histogram are:" 
	# Run the approximate L1 partition algorithm but only ask for the buckets
	print "\t", l1partition.L1partition_approx(x, 1, 0.25, gethist=True)

	numpy.random.seed(10)
	print "The estimated data vector is:"
	# Run the approximate L1 partition algorithm again, and ask for the estimated data vector
	print "\t", l1partition.L1partition_approx(x, 1, 0.25)
	print
	
	print "Compute data- and workload-aware histograms of x using DAWA, with same parameters."	
	print "Workload 1: queries that ask all individual entrires of x."
	
	numpy.random.seed(10)
	print "The estimated data vector is:"	
	Q1 = [ [[1, c, c]] for c in range(len(x)) ]
    #Q1=np.identity(10, dtype=int)
	print "\t", dawa.dawa(Q1, x, 1, 0.25)
	
	print """Workload 2:
	q0 = x0 + x1 + x2
	q1 = x2 + x3 + x4 + x5
	q2 = x3 + x4
	q3 = x5 + x6 + x8 + x9"""
	
	numpy.random.seed(10)
	print "The estimated data vector is:"
	Q2 = [ [ [1, 0, 2] ], \
	[ [1, 2, 5] ], \
	[ [1, 3, 4] ], \
	[ [1, 5, 6], [1, 8, 9] ] ]
	print "\t", dawa.dawa(Q2, x, 1, 0.25)
	print
	
	print "Compute a workload-aware histogram of x by scaling hierarchical queries (the second step of DAWA)."
	numpy.random.seed(10)
	print "The estimated data vector using workload 1 is:"
	print "\t", dawa.dawa(Q1, x, 1, 0)
	
	numpy.random.seed(10)
	print "The estimated data vector using workload 2 is:"
	print "\t", dawa.dawa(Q2, x, 1, 0)
	
	
if __name__ == "__main__":
	main()