import numpy as np
import matplotlib.pyplot as plt
import sys

##########################################################

class dynamics:
    """Newton's eqns of motion and Velocity Verlet algorithms\n
    dt : delta time\n
    m : mass\n
    switch : True for anharmonic potential, False for quartic potential
    """

    def __init__(self, dt, m, switch):
        """initalize dynamics to use delta t and mass"""
        self.dt = dt
        self.m = m
        self.switch = switch

    def vel(self, currVel, frc):
        """Returns the new velocity after a half time-step\n
        currVel : current velocity\n
        frc : current force
        """
        return currVel + (self.dt * frc / 2 / self.m)
    
    def pos(self, currPos, vel, frc):
        """Returns the new position after a full time-step\n
        currPos : current position\n
        vel : current velocity\n
        frc : current force
        """
        return currPos + (vel * self.dt) + (frc * self.dt**2 / 2 / self.m)
    
    def force(self, pos):
        if self.switch: return -pos - 3 * pos**2 / 10 - 4 * pos**3 / 100
        else: return -pos**3
    
    def potE(self, pos):
        """Returns the potential energy at a given position\n
        pos : current positon
        """
        if self.switch: return pos**2 / 2 + pos**3 / 10 + pos**4 / 100
        else: return pos**4 / 4
    
    def kinE(self, vel):
        """Returns kinetic energy at a given velocity\n
        vel : current velocity
        """
        return self.m * vel**2 / 2

########################################################################

class rpmd:
    """Ring polymer molecular dynamics functions\n
    num : number of beads\n
    nsamp : number of samples (trajectories) to generate\n
    scyc : number of cycles per sample\n
    freq : frequency of calculations\n
    beta : inverse T\n
    m : mass of each bead\n
    dt : delta time\n
    switch : True for anharmonic potential, False for quartic potential
    """

    def __init__(self, num, nsamp, neq, scyc, freq, beta, m, dt, switch):
        """Initialalize system constants and arrays to store data"""
        #system constants
        self.num = num
        self.nsamp = nsamp
        self.neq = neq
        self.scyc = scyc
        self.freq = freq
        self.beta = beta
        self.m = m
        self.dt = dt
        self.dyn = dynamics(dt, m, switch)
        if switch: self.name = 'Anharmonic'
        else: self.name = 'Quartic'

        #data arrays
        self.Pos = np.zeros(num)
        self.Vel = np.random.normal(0, np.sqrt(self.num / self.beta / self.m), self.num)
        self.Frc = np.zeros(num)
        self.Cxx = np.zeros([round(scyc / freq), 2])
        self.forces()

    def run(self):
        """Runs the simulation"""
        #initialiazes to NVT
        for i in range(self.neq):
            self.positions()
            self.velocities()
            self.forces()
            self.velocities()
            if i % 50 == 0: self.Vel = np.random.normal(0, np.sqrt(self.num / self.beta / self.m), self.num)
        #loop over 
        for i in range(self.nsamp):
            print('Trajectory number', i + 1, 'for the', self.name, 'potential at beta of', self.beta)
            #resample velocities and initial centroid position
            self.Vel = np.random.normal(0, np.sqrt(self.num / self.beta / self.m), self.num)
            xi = np.mean(self.Pos)
            #MD loop
            for j in range(self.scyc):
                #Velocity Verlet algo
                self.positions()
                self.velocities()
                self.forces()
                self.velocities()
                #check if sample step
                if j % self.freq == 0: 
                    self.calc(j, xi)
                    sys.stdout.flush()
        #final calculation of correlation function
        self.Cxx[:,1] = self.Cxx[:,1] / self.nsamp
            
    
    def forces(self):
        """Updates the forces for all beads"""
        #loop over all beads
        for i in range(self.num):
            #first bead bound to last bead
            if i == 0: self.Frc[i] = -self.m * (self.num / self.beta)**2 * (2 * self.Pos[i] - self.Pos[self.num - 1] - self.Pos[i + 1]) + self.dyn.force(self.Pos[i])
            #last bead bound to first bead
            elif i == self.num - 1: self.Frc[i] = -self.m * (self.num / self.beta)**2 * (2 * self.Pos[i] - self.Pos[i - 1] - self.Pos[0]) + self.dyn.force(self.Pos[i])
            #all other beads bound to adjacent beads
            else: self.Frc[i] = -self.m * (self.num / self.beta)**2 * (2 * self.Pos[i] - self.Pos[i - 1] - self.Pos[i + 1]) + self.dyn.force(self.Pos[i])
    
    def velocities(self):
        """updates the velocities for all beads"""
        self.Vel = self.dyn.vel(self.Vel, self.Frc)
    
    def positions(self):
        """updates the positions of all beads"""
        self.Pos = self.dyn.pos(self.Pos, self.Vel, self.Frc)
    
    def calc(self, step, xi):
        """updates the correlation function\n
        step : current step in simulation\n
        xi = inital centroid position of ring polymer
        """
        self.Cxx[int(step / self.freq), 0] = step * self.dt
        self.Cxx[int(step / self.freq), 1] += np.mean(self.Pos) * xi

def save_data(*data):
    """Generates plots and saves correlation functions and plots\n
    data : data objects stacked vertically
    """
    #generates figure and subplots
    fig, aa = plt.subplots(len(data), sharex=True, figsize=(6.4 * 1.5,3 * 4.8))
    #loops through all data objects
    for i, d in enumerate(data):
        #saves correlation functions to .dat files
        np.savetxt('{0} Cxx Beta {1}.dat'.format(d.name, d.beta), d.Cxx)
        #plot data
        aa[i].plot(d.Cxx[:,0], d.Cxx[:,1])
        #label data
        aa[i].set_title(r'{0} Potential Correlation Function at $\beta = ${1}'.format(d.name, d.beta))
    #label rest of figure
    plt.setp(aa[:], ylabel=r'$C_{xx}(t)$', xlabel='Time')
    for a in aa: a.label_outer()
    #save figure
    plt.tight_layout()
    fig.savefig('Correlation functions.png')

######################################################################################

if __name__ == "__main__":
    nsamp = 1000
    neq = 1000
    scyc = 5000
    freq = 10
    m = 1
    dt = 0.01

    #beta = 1 runs
    beta = 1
    num = 4 * beta
    rpmd1 = rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, True)
    rpmd1.run()
    rpmd2 = rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, False)
    rpmd2.run()

    #beta = 8 runs
    beta = 8
    num = 4 * beta
    rpmd3 = rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, True)
    rpmd3.run()
    rpmd4 = rpmd(num, nsamp, neq, scyc, freq, beta, m, dt, False)
    rpmd4.run()

    save_data(rpmd1, rpmd3, rpmd2, rpmd4)