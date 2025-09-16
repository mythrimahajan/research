#TIME NORMALIZED VERSION OF CASE 3 ANALYTICAL SOLUTIONS 

#now that my code is fixed - can start including time limits 
#divisions, 0, tao, tp, tr 
#exponential decay functions are for t < tao 
#now need to just find tr 

#exponental analytical solution is still growing too slow 
#not able to understand why 

import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.special import erf  


# initial values

alpha1, alpha2 = 1.0, 1.1
b1, b2 = 7.0, 2.0 
u1, u2 = 2.0 ,1.0
c = 1.5
gamma1, gamma2 = 2.5, 1.5
#time
t_range = (0,20)
t_eval = np.linspace(*t_range, 500)
#intensity
I_0 = 0.7
k_0 = 2 * np.pi / 0.8e-6 #based on 800 lambda 
c_light= 3e8
tao = 0.1
tc = 2
#tp=0.175


def intensity(t):
    p1 = 1 + (4 * (t / (k_0 * c_light * tao**2))**2)
    p2 = np.exp(-2 * ((t / tao)**2))
    I = I_0 * p1 * p2
    return I


def alpha2_t(t):
    if (t<0):
        return -1.1
    elif(t<tao):
        return 3.5
    else:
        p3 = np.exp(-(t-tao)/tc)
        return 4.6 * p3 - 1.1


#defines time dependent psi1 and psi2 equations
def psi_time(t, y):
    psi1, psi2= y
    I = intensity(t)
    alpha2_time = alpha2_t(t)
    dpsi1_dt = -gamma1 * ((-alpha1 * psi1) + (4 * u1 * psi1**3) + (2 * c * psi2**2 * psi1) + (b1  * I * psi1))    
    dpsi2_dt = -gamma2 * ((alpha2_time * psi2) + (4 * u2 * psi2**3) + (2 * c * psi1**2 * psi2) + (4 * b2  * I * psi2))
    #switched sign of alpha2 to be consistent 
    return [dpsi1_dt, dpsi2_dt]


#initial values based on paper 
psi1_0_3 = np.sqrt((c * alpha2 - 2 * u2 * alpha1)/(2 * c**2 - 8 * u1 * u2))
psi2_0_3 = np.sqrt((c * alpha1 - 2 * u1 * alpha2)/(2 * c**2 - 8 * u1 * u2))

#solve
sol3 = solve_ivp(psi_time, t_range, [psi1_0_3, psi2_0_3], t_eval=t_eval)
#sol3 = solve_ivp(psi_time, t_range, [psi1_0_3, psi2_0_3],
                # t_eval=t_eval, rtol=1e-9, atol=1e-12)

#values
psi1 = sol3.y[0]
psi2 = sol3.y[1]

#analytical solutions 
#NEED TO PASS IN THE INITIAL SOLUTIONS FOR PSI1 AND PSI2 AS VARIABLES - NOT HARDCODE
#tao to tp 
def analytical_solutions(t):
    t = np.array(t)
    z = np.array(zeta(t))
    p1 = np.exp(-0.25 * gamma1 * b1 * I_0 * tao * z)
    p2 = np.exp(-3.5 * gamma2 * t)
    p3 = np.exp(-gamma2 * b2 * I_0 * tao * z)
    psi1 = psi1_0_3 * p1
    psi2 = psi2_0_3 * p2 * p3
    return psi1, psi2

def zeta(t):
    sigma = k_0 * c_light * tao
    p1 = np.sqrt(2 * np.pi) * (1 + (1/sigma**2)) 
    p2 = erf(np.sqrt(2) * (t/tao))
    p3 = 4 * (1/sigma**2) * (t/tao) * np.exp(-2 * ((t/tao)**2))
    zeta = (p1  * p2) - p3 
    return zeta 

zeta_tao = zeta(tao)
tp = (1/3.5) * ((np.log(40)/gamma2) - (b2 * I_0 * 0.25 * tao * zeta_tao))
print(tp)

idx = np.argmin(np.abs(t_eval - tao))
psi2_tao = psi2[idx]
idx = np.argmin(np.abs(psi2 - psi1_0_3))
tr = t_eval[idx]
idx = np.argmin(np.abs(t_eval-tp))
psi1_tp=psi1[idx]
psi2_tp=psi2[idx]



print(tr)
def phi(t):
    p1 = tc * 4.6
    p2 = np.exp(-(t-tao)/tc) - np.exp(-(tp-tao)/tc)
    p3 = (1.1 * (t - tp))
    return (gamma2 * (p1 * p2 + p3))


def analytical_solutions_2(t):
    #phi_t = phi(t)
    p1 = (4 * u1) - (alpha1/psi1_tp**2)
    p2 = np.exp(-2 * gamma1 * alpha1 * (t-tp))
    p3 = np.exp(phi(t))
    p4 = 4 * u1 * np.exp(2 * gamma1 * alpha1 *(t-tp))
    p5 = (4 * u1 * psi1_tp**2 - alpha1)/(psi1_tp**2)
    p6 = (-gamma2/gamma1)*(c/(4*u1))
    psi1 = np.sqrt(alpha1/(4 * u1 - p1*p2))
    psi2 = (psi2_tp * p3) * ((p4-p5)/(4 * u1 - p5))**p6
    return psi1, psi2


def analytical_solutions_3(t):
    p1 = 4.6 * tc * (np.exp(-(t-tao)/tc) - alpha1)
    psi2 = psi2_tao * np.exp(gamma2 * (1.1 * ((t-tao)) + p1))
    return psi2 


psi1_analytical, psi2_analytical = analytical_solutions(t_eval)
psi1_analytical_2, psi2_analytical_2 = analytical_solutions_2(t_eval)
psi2_analytical_3 = analytical_solutions_3(t_eval)


#CASE 3 psi1 psi2 time evolution plot 
plt.figure(figsize=(10, 6))
plt.plot(sol3.t , sol3.y[0], label='ψ₁')
plt.plot(sol3.t , sol3.y[1], label='ψ₂')

#want these to go until tao, pump on regime - laser on
plt.plot( sol3.t[ (sol3.t   <= tao)  ], psi1_analytical[ (sol3.t   <= tao)  ], "--", label='ψ₁ analytical')
plt.plot( sol3.t[ (sol3.t  <= tao)  ] , psi2_analytical[ (sol3.t   <= tao)  ], "--", label='ψ₂ analytical')

#psi2- pump off regime 
range1 = (sol3.t >= tao) & (sol3.t  <= tp)
plt.plot(sol3.t[range1] , psi2_analytical_3[range1], "--", label = 'ψ₂ analytical')

#metastable state 
range2 = (sol3.t  >= tp) & (sol3.t  <= 13)
plt.plot(sol3.t[range2], psi1_analytical_2[range2], "--", label='ψ₁ analytical')
plt.plot(sol3.t[range2], psi2_analytical_2[range2], "--", label='ψ₂ analytical')



plt.axvline(x = tp  , linestyle = "--", label = 'tp')
plt.axvline(x = tr , linestyle = "--", label = 'tr')
plt.title("Case 3: ψ₁ and ψ₂ v Time")
plt.xlabel("Time")
plt.ylabel("Mean Field Values")
plt.xlim(0,15)
plt.ylim(0,0.5)
plt.legend()
plt.grid(True)
plt.show()
