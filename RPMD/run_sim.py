import RPMD_Verlet_Anharmonic as src

nsamp = 1000
neq = 1000
scyc = 5000
freq = 10
m = 1
dt = 0.01

#beta = 1 runs
beta = 1
num = 4 * beta
rpmd1 = src.rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, True)
rpmd1.run()
rpmd2 = src.rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, False)
rpmd2.run()

#beta = 8 runs
beta = 8
num = 4 * beta
rpmd3 = src.rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, True)
rpmd3.run()
rpmd4 = src.rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, False)
rpmd4.run()

print('Saving Data')
src.save_data(rpmd1, rpmd3, rpmd2, rpmd4)