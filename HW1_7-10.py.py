#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy

RUNS= 1000
iterations_total = 0
mismatch_total = 0

for run in range(RUNS): 
    A = numpy.random.uniform(-1, 1, size = 2)
    B = numpy.random.uniform(-1, 1, size = 2)    
    m = (B[1] - A[1]) / (B[0] - A[0])
    b = B[1] - m * B[0]  
    N = 100
    X = numpy.transpose(numpy.array([numpy.ones(N), numpy.random.uniform(-1, 1, size = N), numpy.random.uniform(-1, 1, size = N)])) 
    woff = numpy.array([b, m, -1])
    X = numpy.transpose(numpy.array([numpy.ones(N), numpy.random.uniform(-1, 1, size = N), numpy.random.uniform(-1, 1, size = N)]))           
    yoff = numpy.sign(numpy.dot(X, woff))                                      
    wofh = numpy.zeros(3)                   
    t = 0                               
    while True:
        yofh = numpy.sign(numpy.dot(X, wofh))   
        comp = (yofh != yoff)                 
        wrong = numpy.where(comp)[0]       
        if wrong.size == 0:
            break
        rnd_choice = numpy.random.choice(wrong)    
        wofh = wofh +  yoff[rnd_choice] * numpy.transpose(X[rnd_choice])
        t += 1

    iterations_total += t
    N_outside = 1000
    test0 = numpy.random.uniform(-1, 1, size = N_outside)
    test1 = numpy.random.uniform(-1, 1, size = N_outside)

    X = numpy.array([numpy.ones(N_outside), test0, test1])

    y_target = numpy.sign(numpy.dot(X, woff))
    y_hypothesis = numpy.sign(numpy.dot(X, wofh))
    
    ratio_mismatch = ((y_target != y_hypothesis).sum()) / N_outside
    ratio_mismatch_total += ratio_mismatch
    
    
print("No. of training data: N = ", N, "points")
    
iterations_avg = iterations_total / RUNS
print("\nAverage number of PLA iterations over", RUNS, "runs: t_avg = ", iterations_avg)

ratio_mismatch_avg = ratio_mismatch_total / RUNS
print("\nAverage ratio for the mismatch between f(x) and h(x) outside of the training data:")
print("P(f(x)!=h(x)) = ", ratio_mismatch_avg)




# In[ ]:





# In[ ]:




