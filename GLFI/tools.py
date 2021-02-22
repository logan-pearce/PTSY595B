import numpy as np
from astropy.time import Time

def update_progress(n,max_value):
    ''' Create a progress bar
    
    Args:
        n (int): current count
        max_value (int): ultimate values
    
    '''
    import sys
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(np.float(n/max_value),decimals=2)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1.:
        progress = 1
        #status = "Done...\r\n"
        status = ''
    block = int(round(barLength*progress))
    text = "\r{0}% ({1} of {2}): |{3}|  {4}".format(np.round(progress*100,decimals=1), 
                                                  n, 
                                                  max_value, 
                                                  "#"*block + "-"*(barLength-block), 
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()


def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def hyperbolic_anomaly(H,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return H - (e*np.sin(H)) - M

def incremental_eccentricity_anomaly(deltaE, deltaM, ecosE, esinE):
    return deltaE + ((1-np.cos(deltaE))*esinE) - (np.sin(deltaE)*ecosE) - deltaM


def incremental_danby_solve(f, deltaM, ecosE, esinE, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly based on "Danby" method in 
        Wisdom textbook
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
        maxnum (int): if it takes more than maxnum iterations,
            use the Mikkola solver instead.
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    import numpy as np
    #f = eccentricity_anomaly
    deltaE0 = deltaM - esinE
    lastE = deltaE0
    nextE = lastE + 10* h 
    number=0
    delta_D = 1
    while (delta_D > h) and number < maxnum+1: 
        fx = f(nextE,deltaM, ecosE, esinE) 
        fp = 1 + np.sin(lastE)*esinE - np.cos(lastE)*ecosE
        fpp = np.cos(lastE)*esinE + np.sin(lastE)*ecosE
        fppp = -np.sin(lastE)*esinE + np.cos(lastE)*ecosE
        lastE = nextE
        delta_N = -fx / fp
        delta_H = -fx / (fp + 0.5*fpp*delta_N)
        delta_D = -fx / (fp + 0.5*fpp*delta_H + (1./6)*fppp*delta_H**2)
        nextE = lastE + delta_D
        number=number+1
        if number >= maxnum:
            from orbittools.orbittools import mikkola_solve
            nextE = mikkola_solve(M0,e)
    return nextE

def solve(f, deltaM, ecosE, esinE, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2019
    '''
    deltaE0 = deltaM - esinE
    lastE = deltaE0
    nextE = lastE + 10* h 
    number=0
    while (abs(lastE - nextE) > h) and number < 1001: 
        fx = f(nextE, deltaM, ecosE, esinE) 
        fp = 1 + np.sin(lastE)*esinE - np.cos(lastE)*ecosE
        lastE = nextE
        nextE = lastE - fx/fp
        number=number+1
        if number >= maxnum:
            nextE = float('NaN')
    return nextE

def hyperbolic_solve(f, M0, e, h):
    ''' Newton-Raphson solver for hyperbolic anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2019
    '''
    import numpy as np
    H0 = M0
    lastH = H0
    nextH = lastH + 10* h 
    number=0
    while (abs(lastH - nextH) > h) and number < 1001: 
        new = f(nextH,e,M0) 
        lastH = nextH
        nextH = lastH - new / (1.-e*np.cos(lastH)) 
        number=number+1
        if number >= 100:
            nextH = float('NaN')
    return nextH

def mikkola_solve(M,e):
    ''' Analytic solver for eccentricity anomaly from Mikkola 1987. Most efficient
        when M near 0/2pi and e >= 0.95.
    Inputs: 
        M (float): mean anomaly
        e (float): eccentricity
    Returns: eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    # Constants:
    alpha = (1 - e) / ((4.*e) + 0.5)
    beta = (0.5*M) / ((4.*e) + 0.5)
    ab = np.sqrt(beta**2. + alpha**3.)
    z = np.abs(beta + ab)**(1./3.)

    # Compute s:
    s1 = z - alpha/z
    # Compute correction on s:
    ds = -0.078 * (s1**5) / (1 + e)
    s = s1 + ds

    # Compute E:
    E0 = M + e * ( 3.*s - 4.*(s**3.) )

    # Compute final correction to E:
    sinE = np.sin(E0)
    cosE = np.cos(E0)

    f = E0 - e*sinE - M
    fp = 1. - e*cosE
    fpp = e*sinE
    fppp = e*cosE
    fpppp = -fpp

    dx1 = -f / fp
    dx2 = -f / (fp + 0.5*fpp*dx1)
    dx3 = -f / ( fp + 0.5*fpp*dx2 + (1./6.)*fppp*(dx2**2) )
    dx4 = -f / ( fp + 0.5*fpp*dx3 + (1./6.)*fppp*(dx3**2) + (1./24.)*(fpppp)*(dx3**3) )

    return E0 + dx4

def kepler_advancer(ro, vo, t, k, to = 0):
    ''' Initial value problem solver using Wisdom-Holman
        numerically well-defined expressions

        Written by Logan A. Pearce, 2020
        
        Parameters:
       -----------
       ro : flt, arr
           initial position vector at time = to; astropy unit object
       vo : flt, arr
           initial velocity vector at time = to; astropy unit object
       t : flt
           future time at which to compute new r,v vectors; 
           astropy unit object
       k : flt
           "Kepler's constant", k = G*(m1+m2); astropy unit object
       to : flt
           initial time for initial values.  Default = 0; 
           astropy unit object
       
       Returns:
       --------
       new_r : flt, arr
           new position vector at time t in m
       new_v : flt, arr
           new velocity vector at time t in m/s
    '''
    import numpy as np
    # Convert everything to mks:
    # Compute magnitude of position vector:
    r = np.linalg.norm(ro)
    # Compute v^2:
    v2 = np.linalg.norm(vo)**2
    # compute u:
    u = ro[0]*vo[0] + ro[1]*vo[1] + ro[2]*vo[2]
    # Compute ang mom h^2:
    h2 = np.linalg.norm(np.cross(ro,vo))**2
    # find a [m] from vis-viva:
    a = (2/r - v2/k)**(-1)
    dt = t - to
    if a > 0:
        # mean motion:
        n = np.sqrt(k/(a**3))
        # ecc:
        esinE = 1 - r/a
        ecosE = u / (n*a*a)
        e = np.sqrt(ecosE*ecosE + esinE*esinE)
        # deltaM:
        deltaM = n*dt
        # E:
        # Delta E:
        deltaE = incremental_danby_solve(incremental_eccentricity_anomaly, deltaM, ecosE, esinE, 1e-6, maxnum=50)
    elif a <= 0:
        a = -a
        # mean motion:
        n = np.sqrt(k/(a**3))
        # ecc:
        esinE = 1 + r/a
        ecosE = u / (n*a*a)
        e = np.sqrt(ecosE*ecosE + esinE*esinE)
        dm = n*dt
        #E0 = np.arctanh(esinE/ecosE)
        E0 = np.log((esinE+ecosE)/e)
        M = esinE - E0 + dm
        E = hyperbolic_solve(hyperbolic_anomaly, M, e, 1e-6)
        if np.isnan(E):
            return np.array([np.nan,np.nan,np.nan]),np.array([np.nan,np.nan,np.nan])
        deltaE = E-E0
    else:
        return np.array([np.nan,np.nan,np.nan]),np.array([np.nan,np.nan,np.nan])
        
    # s2:
    s2 = np.sin(deltaE/2)
    # c2:
    c2 = np.cos(deltaE/2)
    # s:
    s = 2*s2*c2
    # c:
    c = c2*c2 - s2*s2
    # f prime:
    fprime = 1 - c*ecosE + s*esinE
    # f, g:
    f = 1 - 2*s2*s2*a/r
    g = 2*s2*(s2*esinE + c2*r/a)*(1./n)
    # fdot, gdot:
    fdot = -(n*a*s) / (r*fprime)
    gdot = 1 - (2*s2*s2 / fprime)

    # new r:
    new_r = f*ro + g*vo
    # new v:
    new_v = fdot*ro + gdot*vo
    
    return new_r, new_v

def circular_velocity(m,au):
    """ Given separation in AU and total system mass, return the velocity of a test particle on a circular orbit
        around a central body at that mass """
    
    v = np.sqrt(m / au)
    return v

def period(sma,mass):
    """ Given semi-major axis in AU and mass in solar masses, return the period in 2pi years of an orbit using 
        Kepler's third law.
        Written by Logan Pearce, 2021
    """
    period = np.sqrt(((sma)**3)/mass)
    return period

def KeplerEnergy(v, mu, r):
    return 0.5*np.linalg.norm(v)**2 - mu/np.linalg.norm(r)

def Compute_sma_ecc(ro, vo, k):
    ''' Initial value problem solver using Wisdom-Holman
        numerically well-defined expressions

        Written by Logan A. Pearce, 2020
        
        Parameters:
       -----------
       ro : flt, arr
           initial position vector at time = to; 
       vo : flt, arr
           initial velocity vector at time = to; 
       t : flt
           future time at which to compute new r,v vectors; 
       k : flt
           "Kepler's constant", k = G*(m1+m2);
       to : flt
           initial time for initial values.  Default = 0; 
           astropy unit object
       
       Returns:
       --------
    '''
    import numpy as np
    # Convert everything to mks:
    # Compute magnitude of position vector:
    r = np.linalg.norm(ro)
    # Compute v^2:
    v2 = np.linalg.norm(vo)**2
    # compute u:
    u = ro[0]*vo[0] + ro[1]*vo[1] + ro[2]*vo[2]
    # Compute ang mom h^2:
    h2 = np.linalg.norm(np.cross(ro,vo))**2
    # find a [m] from vis-viva:
    a = (2/r - v2/k)**(-1)
    if a > 0:
        # mean motion:
        n = np.sqrt(k/(a**3))
        # ecc:
        esinE = 1 - r/a
        ecosE = u / (n*a*a)
        e = np.sqrt(ecosE*ecosE + esinE*esinE)
    elif a <= 0:
        a = -a
        # mean motion:
        n = np.sqrt(k/(a**3))
        # ecc:
        esinE = 1 + r/a
        ecosE = u / (n*a*a)
        e = np.sqrt(ecosE*ecosE + esinE*esinE)
        a = -a
    
    return a, e


def InitializeOutputFile(initial_pos, initial_vel, initial_post, initial_velt, energy, time, step, dt, output_file = []):
    from astropy.time import Time
    if len(output_file) == 0:
        output_file = 'GLFI-'+str(Time.now().value.year)+\
                        '-'+str(Time.now().value.month)+'-'+str(Time.now().value.day)+'.csv'
    # Create output file and write out initial pos/vel:
    outfile = open(output_file, 'w')
    # Write file header:
    string = '#### Leapfrog Integrator for planet and test particle system '+\
        str(Time.now().value.year)+'-'+str(Time.now().value.month)+'-'+str(Time.now().value.day)+ "\n"
    string += '# initial planet positions: '+','.join(str(p) for p in initial_pos)+ "\n"
    string += '# initial planet velocities: '+','.join(str(p) for p in initial_vel)+ "\n"
    string += '# initial TP positions: '+','.join(str(p) for p in initial_post)+ "\n"
    string += '# initial TP velocities: '+','.join(str(p) for p in initial_velt)+ "\n"
    string += '# dt: '+str(dt)+ "\n"
    outfile.write(string + "\n")
    string = 'step,time,'
    string += 'x,y,z,vx,vy,vz,'
    string += 'xt,yt,zt,vxt,vyt,vzt,energy'
    # Write intial pos/vel
    outfile.write(string + "\n")
    '''
    string = str(step)+','+str(time)+','
    string += ','.join(str(p) for p in initial_pos)
    string += ','
    string += ','.join(str(p) for p in initial_vel)
    string += ','
    string += ','.join(str(p) for p in initial_post)
    string += ','
    string += ','.join(str(p) for p in initial_velt)
    string += ',' + str(energy)'''
    outfile.write(string + "\n")
    outfile.close()

def WriteOutput(pos, vel, post, velt, energy, time, step, output_file = []):
    if len(output_file) == 0:
        output_file = 'GLFI-'+str(Time.now().value.year)+\
                        '-'+str(Time.now().value.month)+'-'+str(Time.now().value.day)+'.csv'
    outfile = open(output_file, 'a')
    string = str(step)+','+str(time)+','
    string += ','.join(str(p) for p in pos)
    string += ','
    string += ','.join(str(p) for p in vel)
    string += ','
    string += ','.join(str(p) for p in post)
    string += ','
    string += ','.join(str(p) for p in velt)
    string += ',' + str(energy)
    outfile.write(string + "\n")
    outfile.close()

class System(object):
    def __init__(self, m1, initial_pos, initial_vel, initial_post, initial_velt, dt = 0.01, Nsteps = [], \
                Norbits = [], output_file = [], run_number = 1, updateprog = True):
        self.initial_pos = initial_pos
        self.initial_vel = initial_vel
        self.initial_r = np.linalg.norm(self.initial_pos)
        self.initial_post = initial_post
        self.initial_velt = initial_velt
        self.initial_rt = np.linalg.norm(self.initial_post)
        if self.initial_rt > self.initial_r:
            self.orbit_type = 'outer'
        if self.initial_rt < self.initial_r:
            self.orbit_type = 'inner'
        # initialize states:
        self.pos = initial_pos
        self.vel = initial_vel
        self.post = initial_post
        self.velt = initial_velt 
        self.mu = 1
        self.m1 = m1
        self.m0 = self.mu - m1
        self.TPenergy = KeplerEnergy(self.velt, self.m0, self.post)
        self.run_number = run_number
        self.dt = dt
        self.updateprog = updateprog
        if np.size(Norbits) != 0:
            self.Norbits = Norbits
            self.Nsteps = np.int_(Norbits/dt)
        elif np.size(Nsteps) != 0:
            self.Nsteps = Nsteps
            self.Norbits = Nsteps*dt
        else:
            print('Must specify either Norbits or Nsteps')

        self.step = 0
        self.t = 0
        if len(output_file) == 0:
            output_file = 'GLFI-'+str(Time.now().value.year)+\
                        '-'+str(Time.now().value.month)+'-'+str(Time.now().value.day)+'-run'+str(run_number)+'.csv'
            self.output_file = output_file
        InitializeOutputFile(self.pos, self.vel, self.post, self.velt, self.TPenergy, self.t, 
                    self.step, self.dt, output_file = self.output_file)

    def Run(self):
        WriteOutput(self.pos, self.vel, self.post, self.velt, self.TPenergy, 
                    self.t, self.step, output_file = self.output_file)
        for j,step in enumerate(range(self.Nsteps)):
            self.step = step
            # Drift: evolve along Keplerian orbits of planet and TP half step:
            # TP: orbit around star only:
            new_post, velt_temp = kepler_advancer(self.post, self.velt, self.dt/2, self.m0)
            # Planet: orbit around total system mass:
            new_pos, new_vel = kepler_advancer(self.pos, self.vel, self.dt/2, self.mu)

            # Apply kicks:
            new_velt = velt_temp.copy()
            denom1 = np.linalg.norm(new_pos)**3
            denom2 = np.sqrt(np.sum((new_post-new_pos)**2))**3
            for i,v in enumerate(velt_temp):
                new_velt[i] = v + self.m1*( (new_post[i] - new_pos[i])/denom1 + (new_pos[i]/denom2) )*self.dt

            # Drift along new keplerian orbits by half a step:
            # TP: orbit around star only:
            new_post2, new_velt2 = kepler_advancer(new_post, new_velt, self.dt/2, self.m0)
            # Planet: orbit around total system mass:
            new_pos2, new_vel2 = kepler_advancer(new_pos, new_vel, self.dt/2, self.mu)

            self.post, self.velt = new_post2, new_velt2
            self.pos, self.vel = new_pos2, new_vel2
            self.t += self.dt
            self.TPenergy = KeplerEnergy(self.velt, self.m0, self.post)
            WriteOutput(self.pos, self.vel, self.post, self.velt, self.TPenergy, 
                    self.t, step, output_file = self.output_file)
            if self.updateprog:
                update_progress(j,self.Nsteps)

            # Check if TP is ejected:
            # Check for hyperbolic orbit:
            if self.TPenergy > 0:
                outfile = open(self.output_file, 'a')
                outfile.write('### Test Particle ejected at step '+str(self.step))
                print('Test Particle ejected at step ',self.step)
                break
            # Check if TP crossed planet's orbit:
            r = np.linalg.norm(new_pos)
            rt = np.linalg.norm(new_post)
            rt2 = np.linalg.norm(new_post2)
            
            if self.orbit_type == 'inner':
                if rt > r or rt2 > r:
                    outfile = open(self.output_file, 'a')
                    outfile.write('### Test Particle crossed planet orbit at step '+str(self.step))
                    print('Test Particle crossed planet orbit at step ',self.step)
                    break
            elif self.orbit_type == 'outer':
                if rt < r or rt2 < r:
                    outfile = open(self.output_file, 'a')
                    outfile.write('### Test Particle crossed planet orbit at step '+str(self.step))
                    print('Test Particle crossed planet orbit at step ',self.step)
                    break
                

    


