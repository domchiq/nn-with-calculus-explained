import numpy as np

"""
what took me ages is understanding that below code is simply derivating final cost function
in this particular case C = (1/2)*(y - yHat)**2 and we are looking for way to minimalize this expression

we use gradient to find the right way - gradient is a vector of partial derivative of C with respect to W1 and
partial derivative of C with respect to W2. this shows us way towards maxima and since we want to
go the opposite direction we subtract it from weights we are currently using to take a step towards minima

please check Siraj Raval video https://www.youtube.com/watch?v=h3l4qz76JhQ&t of code and Welch Labs calculus videos
https://www.youtube.com/watch?v=bxe2T-V8XRs&index=1&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU
"""

def nonlin(x,deriv=False):
	if(deriv==True):
        #sigmoid function derived
	    return x*(1-x)
    #sigmoid function
	return 1/(1+np.exp(-x))
    
X = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])
                
y = np.array([[0],
			[1],
			[1],
			[0]])

np.random.seed(1)

# randomly initialize our weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    k0 = X
    #sigmoid(X*W1)
    k1 = nonlin(np.dot(k0,syn0))
    #sigmoid(k1*W2)
    k2 = nonlin(np.dot(k1,syn1))

    # derived error function - (1/2)*(y - predicted_y)**2
    k2_error = y - k2

    if (j% 10000) == 0:
        #error = mean of all errors
        print ("Error:" + str(np.mean(np.abs(k2_error))))
        
    #derived (1/2)*(y - yHat)**2 multiplied by derivation of sigmoid function applied to
    #output layer - dyHat/dz3 (this is part of derivation of final cost function for both dC/dw1 and dC/dw2)
    k2_delta = k2_error*nonlin(k2,deriv=True)

    # continuing with dC/dw1 using chain rule k2_delta multiplied by dz3/da2
    k1_error = k2_delta.dot(syn1.T)
    
    #another part of dC/dw1 here we derive sigmoid function applied to second layer - da2/dz2
    k1_delta = k1_error * nonlin(k1,deriv=True)
    
    
    #we use + below because dpredicted_y/dz3 is negative but we used positive numbers in multiplication
    
    # final step of derivation dC/dw2 (we multiply k2_delta by dz3/dw2) subtracted from second set of weights
    syn1 += k1.T.dot(k2_delta)
    #final step of derivation dC/dw1 (we multiply k1_delta by dz2/dw1) subtracted from first set of weights
    syn0 += k0.T.dot(k1_delta)