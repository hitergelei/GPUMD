import matplotlib.pyplot as plt



#-------------------作者的ADP的势文件
file = open("Mo.adp", "r")
nrho = 10000
F_rho = []
rho_r = []
z2r = []
u_r = []
w_r = []

i = 0
# 另一种的方法：其实也可以考虑忽略掉前6行，然后直接np.loadtxt()读取整个数据，然后再分别赋值给5个变量列表，这样读取速度应该会更快
for line in file.readlines():  
    i = i + 1

    if i <= 6:   # 前面的6行直接忽略
        pass

    elif i > 6 and i <= 1 * nrho + 6:
        # print(eval(line.split()[0]))
        F_rho.append(eval(line.split()[0]))
    
    elif i > 1 * nrho + 6 and i <= 2 * nrho + 6:
        rho_r.append(eval(line.split()[0]))
    
    elif i > 2 * nrho + 6 and i <= 3 * nrho + 6:
        z2r.append(eval(line.split()[0]))
    
    elif i > 3 * nrho + 6 and i <= 4 * nrho + 6:
        u_r.append(eval(line.split()[0]))
    
    elif i > 4 * nrho + 6 and i <= 5 * nrho + 6:
        w_r.append(eval(line.split()[0]))
    
    else:
        print("------------数据格式长度与目标存取的长度不一致，请检查势文件！")
        exit()

file.close()


#-------------------hjchen的ADP的势文件
file_hj = open("hjchen-Mo.adp", "r")
F_rho_hj = []
rho_r_hj = []
z2r_hj = []
u_r_hj = []
w_r_hj = []

i = 0
# 另一种的方法：其实也可以考虑忽略掉前6行，然后直接np.loadtxt()读取整个数据，然后再分别赋值给5个变量列表，这样读取速度应该会更快
for line in file_hj.readlines():  
    i = i + 1

    if i <= 6:   # 前面的6行直接忽略
        pass

    elif i > 6 and i <= 1 * nrho + 6:
        # print(eval(line.split()[0]))
        F_rho_hj.append(eval(line.split()[0]))
    
    elif i > 1 * nrho + 6 and i <= 2 * nrho + 6:
        rho_r_hj.append(eval(line.split()[0]))
    
    elif i > 2 * nrho + 6 and i <= 3 * nrho + 6:
        z2r_hj.append(eval(line.split()[0]))
    
    elif i > 3 * nrho + 6 and i <= 4 * nrho + 6:
        u_r_hj.append(eval(line.split()[0]))
    
    elif i > 4 * nrho + 6 and i <= 5 * nrho + 6:
        w_r_hj.append(eval(line.split()[0]))
    
    else:
        print("------------数据格式长度与目标存取的长度不一致，请检查势文件！")
        exit()

file_hj.close()

dr = 6.4999999999999997e-04
r_list = [i * dr for i in range(0, 10000)]

#-------------(1) F_rho对比
plt.plot(r_list, F_rho, label="F_rho")
plt.plot(r_list, F_rho_hj, label="F_rho_hj")
plt.legend()
plt.show()
plt.savefig("F_rho.jpg")
plt.close()

#-------------(2) rho_r对比
plt.plot(r_list, rho_r, label="rho_r")
plt.plot(r_list, rho_r_hj, label="rho_r_hj")
plt.legend()
plt.show()
plt.savefig("rho_r.jpg")
plt.close()


#-------------(3) z2r对比
plt.plot(r_list, z2r, label="z2r")
plt.plot(r_list, z2r_hj, label="z2r_hj")
plt.legend()
plt.show()
plt.savefig("z2r.jpg")
plt.close()

#-------------(4) u(r)对比
plt.plot(r_list, u_r, label="u_r")
plt.plot(r_list, u_r_hj, label="u_r_hj")
plt.legend()
plt.show()
plt.savefig("u(r).jpg")
plt.close()

#-------------(5) w(r)对比
plt.plot(r_list, w_r, label="w_r")
plt.plot(r_list, w_r_hj, label="w_r_hj")
plt.legend()
plt.show()
plt.savefig("w(r).jpg")
plt.close()


#-------------(1.2) F_rho与rho的关系对比
drho = 1.0000000000000000e-02
rho_list =  [i * drho for i in range(0, nrho)]
plt.plot(rho_list, F_rho, label="F_rho")
plt.plot(rho_list, F_rho_hj, label="F_rho_hj")
plt.axvline(x=0.85*37.623623, c='r',ls='--',lw=1, label="x=0.85*rhoe")

plt.axvline(x=1.15*37.623623, c='r',ls='--',lw=1, label="x=1.15*rhoe")

plt.legend()
plt.show()
plt.savefig("F_rho_and_rho.jpg")
plt.close()


