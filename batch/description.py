


# Logistic regression for machine learning:
# https://machinelearningmastery.com/logistic-regression-for-machine-learning/

# FM somewhat explained and put into recommender system
# https://dzone.com/articles/factorization-machines-for-recommendation-systems



#Logistic Regression for Machine Learning

# Logistic function (sigmoid function).


# Representation Used for Logistic Regression

# Logistic regression uses an equation as the representation, very much like linear regression.



# Below is an example logistic regression equation:
# y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))





#
#  
#
# L1 Normalization for dummies
# https://medium.com/mlreview/l1-norm-regularization-and-sparsity-explained-for-dummies-5b0e4be3938a
#
#For example, if a vector is [x, y], its L1 norm is |x| + |y|.
# FTL. 
# 
# track the performance of all experts over all previous time steps
# then select the expert/strategy/etc that has performed the best so far
# and follow its advice on the next round
# Update everything and choose again. 
# 
# BTL. Regret
# online convex optimization.
#

https://courses.cs.washington.edu/courses/cse599s/12sp/ :

#
# Proximal-FTRL. aka Follow the proximally-regularized leader
#
#
# Linear model
# Online learning






Deep learning using mini batch (called stochastic gradient descent)
http://neuralnetworksanddeeplearning.com/index.html

An idea called stochastic gradient descent can be used to speed up learning. The idea is to estimate the gradient 
∇C for a small sample of randomly chosen training inputs.

MLP in Python.





http://ruder.io/optimizing-gradient-descent/index.html#gradientdescentvariants




SGD (i.e. online, mini batch size = 1)


Mini-batch



Batch


# On Factorization Machines in general:

https://www.analyticsvidhya.com/blog/2018/01/factorization-machines/


import numpy as np


a = 90
X = [ [5,3,0,1],
      [4,0,0,1],
      [1,1,0,5],
      [1,0,0,4],
      [0,1,5,4]]
    
   
X = np.matrix(X)

# K = 2

P = [[1,2],
     [2,3],
     [1,1],
     [2,4],
     [3,3]]

"""c"""

P = np.matrix(P)

Qt = [[3,5,4,2], [2,3,4,1]]

Qt = np.matrix(Qt)
X2 = P * Qt

X2.shape
X.shape


cdef double[:] w
cdef double[:] z



cdef double predict_single_FM_FTRL(int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
						   double* w, double* z, double* n, double* w_fm, double* z_fm, double* n_fm, double weight_fm,
						   int D_fm, bint bias_term, int threads) nogil:
	cdef int i, ii, k
	cdef double sign, zi, d, wi, wi2, wfmk, e= 0.0, e2= 0.0

	if bias_term:
		if z[0] != 0:
			wi = w[0] = -z[0] / ((beta + sqrt(n[0])) * ialpha)
			e += wi
		else:  w[0] = 0.0

	for ii in prange(lenn, nogil=True, num_threads= threads):
		i= inds[ii]
		zi= z[i]
		sign= -1.0 if zi < 0 else 1.0
		if sign * zi  > L1:
			w[ii+1]= wi= (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2)
			e+= wi * vals[ii]
		else:  w[ii+1] = 0.0

	wi2= 0.0
	for k in prange(D_fm, nogil=True, num_threads=threads):
		wfmk= 0.0
		for ii in range(lenn):
			d= z_fm[inds[ii] * D_fm + k] * vals[ii]
			wfmk= wfmk+d
			wi2+= d **2
		e2+= wfmk **2
		w_fm[k]= wfmk
	e2= (e2- wi2)* 0.5 *weight_fm
	return e+e2

cdef void update_single_FM_FTRL(int* inds, double* vals, int lenn, double e, double ialpha, double* w, double* z, double* n,
						double alpha_fm, double L2_fm, double* w_fm, double* z_fm, double* n_fm,
						int D_fm, bint bias_term, int threads) nogil:
	cdef int i, ii, k
	cdef double g, g2, ni, v, lr, e2= e**2, reg, L2_fme= L2_fm / e
	cdef double *z_fmi
	if bias_term: #Update bias with FTRL-proximal
		g2= e ** 2
		ni= n[0]
		z[0]+= e - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[0]
		n[0]+= g2

	for ii in prange(lenn, nogil=True, num_threads= threads):
	#for ii in range(lenn):
		i= inds[ii]
		v= vals[ii]
		#Update 1st order model with FTRL-proximal
		g= e * v
		g2= g * g
		ni= n[i]
		z[i]+= g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[ii+1]
		n[i]+= g2

		#Update FM with adaptive regularized SGD
		z_fmi= z_fm+ i * D_fm
		lr= g* alpha_fm / (sqrt(n_fm[i])+1.0)
		reg= v - L2_fme
		for k in range(D_fm):  z_fmi[k]-= lr * (w_fm[k] - z_fmi[k] * reg)
		n_fm[i] += e2
        

	def fit_f(self, np.ndarray[double, ndim=1, mode='c'] X_data,
					np.ndarray[int, ndim=1, mode='c'] X_indices,
					np.ndarray[int, ndim=1, mode='c'] X_indptr,
					np.ndarray[double, ndim=1, mode='c'] y,
					sample_weight,
					int threads, int seed):
		cdef double ialpha= 1.0/self.alpha, L1= self.L1, beta= self.beta, baL2= beta * ialpha + self.L2, \
					alpha_fm= self.alpha_fm, weight_fm= self.weight_fm, L2_fm= self.L2_fm, e, e_total= 0, zfmi, \
					e_noise= self.e_noise, e_clip= self.e_clip, abs_e
		cdef double *w= &self.w[0], *z= &self.z[0], *n= &self.n[0], *n_fm= &self.n_fm[0], \
					*z_fm= &self.z_fm[0], *w_fm= &self.w_fm[0], *ys= <double*> y.data
		cdef unsigned int D_fm= self.D_fm, lenn, ptr, row_count= X_indptr.shape[0]-1, row, inv_link= self.inv_link
		cdef bint bias_term= self.bias_term
		cdef int* inds, indptr
		cdef double* vals

		rand = randomgen.xoroshiro128.Xoroshiro128(seed=seed).generator

		#For iter goes here
		e_total= 0.0
		for row in range(row_count):
			ptr = X_indptr[row]
			lenn = X_indptr[row+1]-ptr
			
            # Sparse row data
            inds= <int*> X_indices.data+ptr
			vals= <double*> X_data.data+ptr

            p = predict_single_FM_FTRL(inds, vals, lenn,
											L1, baL2, ialpha, beta, w, z, n,
											w_fm, z_fm, n_fm, weight_fm,
											D_fm, bias_term, threads)

            e = p - ys[row]

            e_total+= abs_e
            # optional noise, weight, clip on e
                                                
            update_single_FM_FTRL(inds, vals, lenn, e, ialpha, w, z, n, alpha_fm, L2_fm, w_fm, z_fm, n_fm, D_fm,
								bias_term, threads)

		print "Total e:", e_total
		return self


###############################   FTRL   #####################################################

cdef double[:] w   init zero
cdef double[:] z   init zero
cdef double[:] n   init zero


cdef double FTRL_predict_single(int* inds, double* vals, int lenn, double L1, double baL2, double ialpha, double beta,
				double[:] w, double[:] z, double[:] n, bint bias_term, int threads) nogil:

	cdef int i, ii
	cdef double sign, zi, wi
	cdef double e= 0.0
	if bias_term:
		if z[0] != 0:
			wi = w[0] = -z[0] / ((beta + sqrt(n[0])) * ialpha)
			e += wi
		else:  w[0] = 0.0

	for ii in prange(lenn, nogil=True, num_threads= threads):
		i= inds[ii]
		zi= z[i]
		sign = -1.0 if zi < 0 else 1.0
		if sign * zi  > L1:
			wi= w[i] = (sign * L1 - zi) / (sqrt(n[i]) * ialpha + baL2)
			e+= wi * vals[ii]
		else:  w[i] = 0.0
	return e

cdef void FTRL_update_single(int* inds, double* vals, int lenn, double e, double ialpha, double[:] w, double[:] z,
						double[:] n, bint bias_term, int threads) nogil:
	cdef int i, ii
	cdef double g, g2, ni
	if bias_term:
		g2= e ** 2
		ni= n[0]
		z[0]+= e - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[0]
		n[0]+= g2

	for ii in prange(lenn, nogil=True, num_threads= threads):
		i= inds[ii]
		g= e * vals[ii]
		g2= g ** 2
		ni= n[i]
		z[i]+= g - ((sqrt(ni + g2) - sqrt(ni)) * ialpha) * w[i]
		n[i]+= g2



	def FTRL_fit_f( X_data, X_indices,X_indptr, y):
		double ialpha= 1.0/self.alpha,
        double baL2= beta * ialpha + self.L2,
        double e_total= 0
        
		cdef double[:] w= self.w, z= self.z, n= self.n, 
         
        ys= y

		cdef unsigned int lenn, ptr, row_count= X_indptr.shape[0]-1, row, inv_link= self.inv_link, j=0, jj
		
        cdef bint bias_term= self.bias_term
		
        cdef int* inds, indptr
		cdef double* vals

		for iter in range(self.iters):
			e_total= 0.0
			for row in range(row_count):
				ptr= X_indptr[row]
				lenn= X_indptr[row+1]-ptr
				inds= <int*> X_indices.data+ptr
				vals= <double*> X_data.data+ptr
				e = FTRL_predict_single(inds, vals, lenn, L1, baL2, ialpha, beta, w, z, n, bias_term)-ys[row]

                e_total+= fabs(e)
				
                # operations on e clip, invert et.c.

				FTRL_update_single(inds, vals, lenn, e, ialpha, w, z, n, bias_term, threads)

			if self.verbose > 0:  print "Total e:", e_total
		return self
