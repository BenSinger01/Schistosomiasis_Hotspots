import numpy as np
from scipy import stats
import sys


# p, n, p_value = sys.argv[1], sys.argv[2], sys.argv[3]

def interval(p,n,p_value):
	z = stats.norm.ppf(1-(1-p_value)/2)
	mid = (1/(1+(z**2)/n))*(p + (z**2)/(2*n))
	pom = (z/(1+(z**2)/n))*np.sqrt((p*(1-p))/n + (z**2)/(4*(n**2)))
	return((mid-pom,mid+pom))