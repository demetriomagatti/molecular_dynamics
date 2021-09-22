import numpy as np
from scipy.spatial.distance import cdist
import scipy
import pylab as pp


#constant values 
rc = 4.625
rp = 4.375
sigma = 2.644
epsilon = 0.345
kb = 1/11603
Temp = 100
m_ag = 108*1.66e-27/16


#lennard jones poly7 approximation coefficients
Apoli7 = (1/((rc - rp)**7*rp**12))*4*epsilon*rc**4*sigma**6*(2* rp**6 *(-42* rc**3 + 182* rc**2* rp - 273* rc* rp**2 + 143* rp**3) + (455* rc**3 - 1729* rc**2* rp + 2223* rc* rp**2 - 969* rp**3)*sigma**6)
Bpoli7 = (1/((rc - rp)**7* rp**13))*16* epsilon* rc**3* sigma**6*(rp**6* (54* rc**4 - 154* rc**3* rp + 351* rc *rp**3 - 286* rp**4) + (-315* rc**4 + 749 *rc**3 * rp + 171 *rc**2* rp**2 - 1539* rc* rp**3 + 969* rp**4)* sigma**6)
Cpoli7 = (1/((rc - rp)**7* rp**14))*12* epsilon* rc**2* sigma**6* (rp**6* (-63* rc**5 - 7* rc**4 *rp + 665* rc**3 *rp**2 - 975* rc**2* rp**3 - 52* rc* rp**4 + 572* rp**5) +  2 *(195* rc**5 + 91* rc**4* rp - 1781* rc**3* rp**2 + 1995 *rc**2* rp**3 + 399* rc* rp**4 - 969* rp**5)* sigma**6)
Dpoli7 = (1/((rc - rp)**7* rp**15))*16* epsilon* sigma**6*(rc* rp**6* (14* rc**6 + 126* rc**5* rp - 420* rc**4* rp**2 - 90* rc**3* rp**3 + 1105* rc**2* rp**4 - 624* rc* rp**5 - 286 *rp**6) + rc* (-91* rc**6 - 819* rc**5* rp + 2145 * rc**4 * rp**2 + 1125* rc**3* rp**3 - 5035* rc**2* rp**4 + 1881* rc* rp**5 + 969* rp**6)* sigma**6)
Epoli7 = (1/((rc - rp)**7* rp**15))*4* epsilon* sigma**6*(2* rp**6* (-112* rc**6 - 63* rc**5* rp + 1305* rc**4* rp**2 - 1625* rc**3* rp**3 - 585* rc**2* rp**4 +  1287 *rc* rp**5 + 143* rp**6) + (1456*rc**6 +1404*rc**5* rp - 14580 *rc**4* rp**2 + 13015* rc**3* rp**3 + 7695* rc**2* rp**4 - 8721 *rc* rp**5 - 969* rp**6)* sigma**6)
Fpoli7 = (1/((rc - rp)**7* rp**15))*48* epsilon* sigma**6*(-rp**6* (-28* rc**5 + 63* rc**4* rp + 65* rc**3* rp**2 - 247* rc**2* rp**3 + 117* rc* rp**4 + 65* rp**5) + (-182* rc**5 + 312* rc**4* rp + 475* rc**3* rp**2 - 1140* rc**2* rp**3 + 342* rc* rp**4 + 228* rp**5)* sigma**6)
Gpoli7 = (1/((rc - rp)**7* rp**15))*4* epsilon* sigma**6* (rp**6* (-224* rc**4 + 819* rc**3* rp - 741* rc**2* rp**2 - 429* rc* rp**3 + 715* rp**4) + 2 *(728* rc**4 - 2223* rc**3* rp + 1425* rc**2* rp**2 + 1292* rc* rp**3 - 1292* rp**4)* sigma**6)
Hpoli7 = (1/((rc - rp)**7* rp**15))*16* epsilon* sigma**6* (rp**6*(14* rc**3 - 63* rc**2* rp + 99* rc* rp**2 - 55* rp**3) + (-91* rc**3 + 351* rc**2* rp - 459* rc* rp**2 + 204* rp**3)* sigma**6)


#total potential energy
def lennard_jones():
    '''
    returns total potential energy value
    '''
    return 