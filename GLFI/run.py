from tools import *

# Masses
mu = 1
# planet:
logm1 = np.linspace(-5.5,-2,8)
m1 = 10**logm1
# star:
m0 = mu-m1

initial_seps = np.arange(0.7,1.0,0.01)
initial_seps = initial_seps[::-1]
print(initial_seps)

for j,mass in enumerate(m1):
    count = 0
    for i,sep in enumerate(initial_seps):
        print('Mass =',np.log10(m1[j]),':',j+1,' of ',len(m1)+1,': Testing ',i+1,' of ',len(initial_seps)+1, 'sep = ',np.round(sep, decimals=2))
        # Initial heliocentric positions:
        # Planet:
        x, y, z = 1, 0, 0 # AU
        vx, vy, vz = 0, 1, 0 # AU/2piyr
        # Test particle heliocentric:
        xt, yt, zt = sep, 0, 0 # AU
        vxt, vyt, vzt = 0, circular_velocity(m0[j],xt), 0 # AU/2piyr

        post = np.array([xt,yt,zt])
        velt = np.array([vxt,vyt,vzt])
        pos = np.array([x,y,z])
        vel = np.array([vx,vy,vz])
        r0t = np.linalg.norm(post)
        dt = period(r0t,m0[j])/10
        Norbits = 1e4

        system = System(m1[j], pos, vel, post, velt, dt = dt, Norbits = Norbits, 
                        run_number = '-m'+str(np.abs(np.log10(m1[j])))+'-sep'+str(np.round(sep, decimals=2)))
        system.Run()
        # If the intgrator finds three stable orbits in a row, move on to the next mass:
        steps_completed = system.step+1
        if steps_completed == system.Nsteps:
            count += 1
        if steps_completed != system.Nsteps:
            count = 0
        print('Steps completed',steps_completed, 'count=',count)
        if count == 4:
            print('2 stable orbits in a row, moving on')
            break
        
