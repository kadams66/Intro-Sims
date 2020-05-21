import numpy as np
import matplotlib.pyplot as plt

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
        """Returns the new velocity after a half time-step.\n
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

    def __init__(self, num, nsamp, scyc, freq, beta, m, dt, switch):
        """Initialalize system constants and arrays to store data"""
        #system constants
        self.num = num
        self.nsamp = nsamp
        self.scyc = scyc
        self.freq = freq
        self.beta = beta
        self.m = m
        self.dt = dt
        self.dyn = dynamics(dt, m, switch)
        self._ncalc = 0
        if switch: self.name = 'Anharmonic'
        else: self.name = 'Quartic'

        #data arrays
        self.Pos = np.zeros(num)
        self.Vel = np.zeros(num)
        self.Frc = np.zeros(num)
        self.Cxx = np.zeros([round(scyc / freq), 2])
        self.forces()

    def run(self):
        """Runs the simulation"""
        #loop over 
        for i in range(self.nsamp):
            print('Trajectory number ', i + 1, ' for the', self.name, ' potential' )
            #resample velocities and initial centroid position
            self.Vel = np.random.normal(0, np.sqrt(1 / beta / m), self.num)
            xi = np.mean(self.Pos)
            #MD loop
            for j in range(self.scyc):
                #Velocity Verlet algo
                self.positions()
                self.velocities()
                self.forces()
                self.velocities()
                #check if sample step
                if j % self.freq == 0: self.calc(j, xi)
        #final calculation of correlation function
        self.Cxx[:,1] = self.Cxx[:,1] / self._ncalc
            
    
    def forces(self):
        """Updates the forces for all beads"""
        #loop over all beads
        for i in range(self.num):
            #first bead bound to last bead
            if i == 0: self.Frc[i] = -self.m / self.beta**2 * (2 * self.Pos[i] - self.Pos[self.num - 1] - self.Pos[i + 1]) + self.dyn.force(self.Pos[i])
            #last bead bound to first bead
            elif i == self.num - 1: self.Frc[i] = -self.m / self.beta**2 * (2 * self.Pos[i] - self.Pos[i - 1] - self.Pos[0]) + self.dyn.force(self.Pos[i])
            #all other beads bound to adjacent beads
            else: self.Frc[i] = -self.m / self.beta**2 * (2 * self.Pos[i] - self.Pos[i - 1] - self.Pos[i + 1]) + self.dyn.force(self.Pos[i])
    
    def velocities(self):
        """updates the velocities for all beads"""
        self.Vel = self.dyn.vel(self.Vel, self.Frc)
    
    def positions(self):
        """updates the positions of all beads"""
        self.Pos = self.dyn.pos(self.Pos, self.Vel, self.Frc)
    
    def calc(self, step, xi):
        """updates the correlation function\n
        step : current step in simulation
        xi = inital centroid position of ring polymer
        """
        self._ncalc += 1
        self.Cxx[int(step / self.freq), 0] = step * self.dt
        self.Cxx[int(step / self.freq), 1] += np.mean(self.Pos) * xi

def save_data(data1, data2):
    """Generates plots and saves correlation functions and plots\n
    data1 : first object
    data2 : second object
    """
    #saves correlation functions to .dat files
    np.savetxt('Anharmonic Cxx.dat', data1.Cxx)
    np.savetxt('Quartic Cxx.dat', data2.Cxx)
    #generates figure and subplots
    fig, aa = plt.subplots(2, sharex=True)
    #plot data
    aa[0].plot(data1.Cxx[:,0], data1.Cxx[:,1])
    aa[1].plot(data2.Cxx[:,0], data2.Cxx[:,1])
    #label figure
    plt.setp(aa[:], ylabel=r'$C_{xx}(t)$', xlabel='Time')
    aa[0].set_title('Anharmonic Potential Correlation Function')
    aa[1].set_title('Quartic Potential Correlation Function')
    for a in aa: a.label_outer()
    #save figure
    fig.savefig('Correlation functions.png')

######################################################################################

if __name__ == "__main__":
    nsamp = 100
    scyc = 1000
    freq = 5
    m = 1
    dt = 0.05
    beta = 1
    num = int(4 * beta)

    rpmd1 = rpmd(num, nsamp, scyc, freq, beta, m, dt, True)
    rpmd1.run()

    rpmd2 = rpmd(num, nsamp, scyc, freq, beta, m, dt, False)
    rpmd2.run()

    save_data(rpmd1, rpmd2)