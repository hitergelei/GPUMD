import numpy as np
from datetime import datetime




#-----rho(r), phi(r), F(r) 参数----------
rho_phi_F_param = \
{'Mo':
    {
    're':   2.7281,
    'fe':    3.5863051,
    'rhoe':  37.623623,
    'rhos':  22.683228,
    'alpha': 7.6616936,
    'beta':  5.5784864,
    'A':     0.9215712,
    'B':     1.7317773,
    'kappa': 0.1413604,
    'lambda':0.24908023,
    'Fn0':   -6.270608,
    'Fn1':   2.2659059,
    'Fn2':   -0.18881902,
    'Fn3':   -3.2595265,
    'F0':    -5.8015256,
    'F1':    3.2561238,
    'F2':    1.1035414,
    'F3':    -0.95508283,
    'eta':   0.7645085,
    'Fe':    -6.360732
    }

}




dipole_param = \
{'Mo':
    {
    'd1': -0.10194129,
    'd2':  -2.098797,
    'd3': 6.1936436, 
    'rc': 4.4990587,
    'h' : 6.757866
    }

}

quadrupole_param = \
{'Mo':
    {
    'q1': 0.08105006,
    'q2': -1.6661074,
    'q3': -9.597149,
    'rc': 4.4990587,
    'h' : 6.757866
    }

}






para =  rho_phi_F_param['Mo']


# 《Machine learning enhanced empirical potentials for metals and alloys》
#----F(rho)---eq (18)
def emb(rho):
    rho_e = para['rhoe']
    Fn0 = para['Fn0']
    Fn1 = para['Fn1']
    Fn2 = para['Fn2']
    Fn3 = para['Fn3']
    F0 = para['F0']
    F1 = para['F1']
    F2 = para['F2']
    F3 = para['F3']
    Fe = para['Fe']
    eta = para['eta']
    rho_s = para['rhos']
    

    rho_n = 0.85 * rho_e   # rho_n = 31.98007955
    rho_0 = 1.15 * rho_e   # rho_0 = 43.26716645

    if rho < rho_n:
        x1 = rho / rho_n - 1
        x2 = x1**2
        x3 = x1**3
        e10 = Fn0 
        e11 = Fn1 * x1
        e12 = Fn2 * x2
        e13 = Fn3 * x3
        return e10 + e11 + e12 + e13
    
    elif rho < rho_0 and rho >= rho_n:  # rho_n <= rho < rho_0
        t = rho / rho_e - 1  # t = (ρ / ρe - 1)
        x0 = t**0
        x1 = t**1
        x2 = t**2
        x3 = t**3
        e10 = F0 * x0   # F0 = para['F0']
        e11 = F1 * x1
        e12 = F2 * x2
        e13 = F3 * x3
        return e10 + e11 + e12 + e13
    
    else:  # rho >= rho_0
        x = rho / rho_s
        lnx = np.log(x)   # ln(rho / rho_s)
        return Fe * (1 - eta * lnx) * (x)**eta




#-------------Zhou, Johnson and Wadley (zjw04) EAM-------------
def rho(r):
    fe = para['fe']
    beta = para['beta']
    re = para['re']
    lambda_ = para['lambda']

    term1 = fe * np.exp(-beta *  (r / re - 1))
    term2 = 1 + (r / re - lambda_)**20   # 使用np.power()速度更快? 
    return term1 / term2

#------Mo的φ(r)
def phi_AA(r):
    A = para['A']
    re = para['re']
    alpha = para['alpha']
    kappa = para['kappa']
    B = para['B']
    beta = para['beta']
    lambda_ = para['lambda']

    term1 = A * np.exp(-alpha * (r / re - 1))
    term2 = 1 + (r / re - kappa)**20
    left = term1 / term2

    term3 = B * np.exp(-beta * (r / re - 1))
    term4 = 1 + (r/ re - lambda_)**20
    right = term3 / term4
    return left - right



# 《Angular-dependent interatomic potential for the aluminum-hydrogen system》

# ψ(r)-cutoff funciton  eq (7)
def psi(x):
    if x < 0:
        return x**4 / (1 + x**4)
    else:
        return 0




#----u(r) - dipole funcitons  eq (11)
def u11(r):
    d1= dipole_param['Mo']['d1']
    d2 = dipole_param['Mo']['d2']
    d3 = dipole_param['Mo']['d3']
    rc = dipole_param['Mo']['rc']
    h = dipole_param['Mo']['h']
    assert rc == quadrupole_param['Mo']['rc']
    assert h == quadrupole_param['Mo']['h']

    # term1 = d1 * np.exp(-d2) + d3   #  Al-H论文和其他论文中公式不同，可能是作者d2后面漏乘了r， 此外quadrupole中也是漏乘了r
    # 比如在《Angular-dependent interatomic potential for tantalum》公式中就有r项。https://doi.org/10.1016/j.actamat.2006.06.034
    term1 = d1 * np.exp(-d2 * r) + d3
    x = (r - rc) / h
    term2 = psi(x)
    return term1 * term2
    


#-----w(r) - quadrupole functions eq (12)
def w11(r):
    q1 = quadrupole_param['Mo']['q1']
    q2 = quadrupole_param['Mo']['q2']
    q3 = quadrupole_param['Mo']['q3']
    rc = quadrupole_param['Mo']['rc']
    h  = quadrupole_param['Mo']['h']

    assert rc == dipole_param['Mo']['rc']
    assert h  == dipole_param['Mo']['h']

    # term1 = q1 * np.exp(-q2) + q3   # 该公式写法有问题，弃用
    term1 = q1 * np.exp(-q2 * r) + q3
    x = (r - rc) / h
    term2 = psi(x)
    return term1 * term2


if __name__ == "__main__":
    n_elements = 1
    atom_type = 'Mo'

    nrho = 10000; drho = 1.0000000000000000e-02; 
    nr = 10000;   dr = 6.4999999999999997e-04 
    cutoff = 6.5000000000000000e+00

    automic_number = 42; mass = 9.5950000000000003e+01
    lattice_constant = 3.1499999999999999e+00; lattice_type = 'bcc'
 
    file = open('hjchen-Mo.adp', "w")
    file.write("writen by hjchen in %s\n" % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    file.write("LAMMPS setfl format for Mo element adp potential\n")
    file.write("Convert by python language\n")
    file.write("{} {}\n".format(n_elements, atom_type))
    file.write("{} {:.16e} {} {:.16e} {:.16e}\n".format(nrho, drho, nr, dr, cutoff))
    file.write("{} {:.16e} {:.16e} {}\n".format(automic_number, mass, lattice_constant, lattice_type))


    #----------------(1) 存储ADP势文件中需要的变量数组的顺序内容： F(rho), rho(r), r*Φ(r), w(r), u(r)
    emb_list = []
    r = 0
    for i in range(0, nrho):
        rho_value =  i * drho
        # print(emb(rho))
        # exit(0)
        r = i * dr
        if r == 2.08:
            print("r = ", r)
            print("rho_value = ", rho_value)
            print("emb(rho_value) = ", emb(rho_value))
            # break
        emb_list.append(emb(rho_value))
    
   
    rho_list = []
    for i in range(0, nr):
        r = i * dr
        rho_list.append(rho(r))
    

    # 即rij * φ(rij)的值
    z2r_list = []
    for i in range(0, nr):
        r = i * dr
        z2r_list.append(r * phi_AA(r))
    
    u2r_list = []  
    for i in range(0, nr):
        r = i * dr
        u2r_list.append(u11(r))
    
    w2r_list = []  # w(r)
    for i in range(0, nr):
        r = i * dr
        w2r_list.append(w11(r))
    

    #----------------(2) 写入势文件中
    for i in range(0, nr):
        file.write("{:.16e}\n".format(emb_list[i]))

    for i in range(0, nrho):
        file.write("{:.16e}\n".format(rho_list[i]))
    
    for i in range(0, nr):
        file.write("{:.16e}\n".format(z2r_list[i]))

    for i in range(0, nr):
        file.write("{:.16e}\n".format(u2r_list[i]))

    for i in range(0, nr):
        file.write("{:.16e}\n".format(w2r_list[i]))

    file.close()













