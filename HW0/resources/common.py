import numpy as np
CONV_TESTS = False

if CONV_TESTS:
    from scipy.signal import convolve2d

######
#shortcuts
gauss = lambda t: np.random.randn(*t)
unif = lambda t: np.random.uniform(size=t)
unifint = lambda l,h,t: np.random.randint(l,h+1,size=t)
if CONV_TESTS:
    conv2dfull = lambda I,f: convolve2d(I,f,mode='full')
    conv2dvalid = lambda I,f: convolve2d(I,f,mode='valid')
    conv2dsame = lambda I,f: convolve2d(I,f,mode='same')

def frank(n=9,minSV=1,psd=True):
    #return a full-rank nxn matrix with varying svs
    M = unif((n,n))
    if psd: M = np.dot(M.T,M)
    U,S,VT = np.linalg.svd(M)
    for i in range(n):
        S[i] += (minSV + 1.0 / (i+1))
    return np.dot(np.dot(U,np.diag(S)),VT)

def rot(n=3):
    M = unif((n,n))
    U,S,VT = np.linalg.svd(M)
    R = np.dot(U,VT)
    return np.linalg.det(R) * R
######


#####
#Test-case generators
gen_b1 = lambda: gauss((10,10))
gen_b2 = lambda: (gauss((10,10)),gauss((10,10)))
gen_b3 = lambda: (gauss((10,10)),gauss((10,10)))
gen_b4 = lambda: (gauss((10,10)),gauss((10,10)))
gen_b5 = lambda: np.round(unif((5,5))*100)
gen_b6 = lambda: (unifint(1,100,(10,1)),np.ones((10,1),dtype=np.int)*100)
gen_b7 = lambda: unif((10,20))
gen_b8 = lambda: np.random.choice(20)+30
gen_b9 = lambda: unif((10,10))
gen_b10 = lambda: np.random.choice(10)+10
gen_b11 = lambda: (gauss((10,20)), gauss((20,1)))
gen_b12h = lambda s: (frank(n=s),gauss((s,1))) #sigh
gen_b12 = lambda: gen_b12h(np.random.choice(10)+5)
gen_b13h = lambda s: (gauss((s,1)),gauss((s,1)))
gen_b13 = lambda: gen_b13h(np.random.choice(10)+5)
gen_b14 = lambda: gauss((np.random.choice(10)+5,1))
gen_b15h = lambda s: (gauss((s,3*s)),np.random.choice(s))
gen_b15 = lambda: gen_b15h(np.random.choice(10)+5)
gen_b16h = lambda s: gauss((s,s+2))
gen_b16 = lambda: gen_b16h(np.random.choice(4)+2)
gen_b17 = gen_b16
gen_b18 = gen_b16
gen_b19 = gen_b16
gen_b20 = gen_b16

gen_p1 = lambda: [unif((1,4)) for i in range(10)]
gen_p2 = frank
gen_p3 = lambda: unif((4,3))*2-1
gen_p4 = lambda: (rot(), unif((10,3)))
gen_p5 = lambda: unif((100,100))
gen_p6 = lambda: np.random.choice(20)+20
gen_p7 = lambda: gauss((10,10))
gen_p8 = lambda: gauss((10,10))
gen_p9 = lambda: (gauss((1,10)), gauss((100,10)), gauss((100,1)))
gen_p10 = lambda: [gauss((10,4)) for i in range(30)]
gen_p11 = lambda: gauss((100,4))
gen_p12 = lambda: (gauss((100,4)),gauss((200,4)))
gen_p13 = lambda: (gauss((1,4)), gauss((30,4)))
gen_p14 = lambda: (gauss((100,10)), gauss((100,1)))
gen_p15 = lambda: (gauss((100,3)),gauss((100,3)))
gen_p16 = lambda: gauss((100,3))
gen_p17 = lambda: gauss((100,4))
gen_p18 = lambda: (100,10,np.random.choice(80)+10,np.random.choice(80)+10)
gen_p19 = lambda: (100,10,np.random.choice(80)+10,np.random.choice(80)+10)
gen_p20 = lambda: (100,np.array([1,-1,0]))


gen_c1 = lambda: (gauss((10,10)),3)
gen_c2 = lambda: (gauss((10,10)),3)
gen_c3 = lambda: (np.ones((12,12)),2)
gen_c4 = lambda: (gauss((100,100)),7)
#####

