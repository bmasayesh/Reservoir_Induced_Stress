# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This code calculate stress du to water impoundment in the Gotvand Dam (SW Iran)
'''

########################## Importing Required Modules #########################
import sys
import math
import datetime as dt
from datetime import datetime
import numpy as np
year2sec = 365.25*24.0*60*60
density = 1000.0   # density of water [kg/m^3] 
gravity = 9.81     # [m/s^2]

############################# Functions & Classes #############################
## Function for calculating time in decimal ===================================
def timeyear(year, month, day, hour, minute, sec):
    time = np.zeros(len(year))
    for i in range(len(year)):
        events_datetime =  dt.datetime(int(year[i]), int(month[i]), 
                             int(day[i]), int(hour[i]), 
                             int(minute[i]), int(sec[i]))
        events_time = datetime.timestamp(events_datetime)
        events_datetime1 =  dt.datetime(int(year[i]), 1, 1, 0, 0, 0)
        events_datetime2 =  dt.datetime(int(year[i]+1), 1, 1, 0, 0, 0)
        T1 = datetime.timestamp(events_datetime1)
        T2 = datetime.timestamp(events_datetime2)
        time[i] = year[i] + (events_time-T1)/(T2-T1)
    return time

## Function for Calculation of Distance in 2D =================================
def dist(lat1, lon1, lat2, lon2):
    """
    Distance (in km) between points given as [lat,lon]
    """
    R0 = 6367.3        
    D = R0 * np.arccos(
            np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) +
            np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2)))
    return D

def dist_all(lat1, lon1, lat2, lon2):
    """
    Distance (in km) between points given as [lat,lon]
    """
    R0 = 6367.3
    arg = np.sin(np.radians(lat1)) * np.sin(np.radians(lat2)) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon1-lon2))
    R = np.zeros(len(arg))
    R[(arg<1)] = R0 * np.arccos(arg[(arg<1)])
    return R 

## Function for calculation of stress matrix based on Deng et al., BSSA 2010 ==
def dengStresstensor(fac, strike, lat, lon, z, latsource, lonsource):
    # Deng et al., BSSA 2010: Eq.(6) & (7):
    # local coordinate system: x-axis in strike direction and z-axis pointing down (i.e. y-axis in relativ east-direction)
    # origin: source point
    # constants:
    a  = 1.0 - 2*poisson;
    ss = strike * np.pi/180.0;
    
    # distance to source [m]:
    rho  = 1000.0 * dist(lat,lon,latsource,lonsource)
    xlat = 1000.0 * dist(latsource,lonsource,lat,lonsource)
    ylon = np.sqrt( np.square(rho) - np.square(xlat) )
    xlat[(lat<latsource)] *= -1.0
    ylon[(lon<lonsource)] *= -1.0
    
    # rotation of coordinate system into strike direction:
    # test: point located in strike direction: xlat=rho*cos(strike); ylon=rho*sin(strike);
    #       rotation must lead to x=rho  and y=0 
    x = np.sin(ss)*ylon + np.cos(ss)*xlat
    y = np.cos(ss)*ylon - np.sin(ss)*xlat
    R = np.sqrt( np.square(rho) + np.square(z) )
    
    phi  = np.arcsin(y/rho)
    
    Srho = (1.0/(2.0*np.pi*np.square(R))) * ( a*R/(R+z) - 3.0*np.square(rho)*z/np.power(R,3.0) )
    Sphi =   (a/(2.0*np.pi*np.square(R))) * ( z/R - R/(R+z) )
    
    Sxx  = Srho * np.square(np.cos(phi)) + Sphi * np.square(np.sin(phi))
    Syy  = Srho * np.square(np.sin(phi)) + Sphi * np.square(np.cos(phi))
    Szz  = - 3.0 * np.power(z,3.0) / (2.0 * np.pi * np.power(R,5.0))
    
    Sxy  = (Srho - Sphi) * np.sin(phi) * np.cos(phi);
    Syz  = - 3.0 * rho * np.square(z) * np.sin(phi) / (2.0 * np.pi * np.power(R,5.0))
    Szx  = - 3.0 * rho * np.square(z) * np.cos(phi) / (2.0 * np.pi * np.power(R,5.0))
    
    sxx = fac * Sxx;
    syy = fac * Syy;
    szz = fac * Szz;
    sxy = fac * Sxy;
    syz = fac * Syz;
    szx = fac * Szx;
    return sxx, syy, szz, sxy, szx, syz

## Function for calculation of Coulomb stress changes =========================         
def dengCFS(dip, rake, friction, skempton, sxx, syy, szz, sxy, szx, syz):
    # Deng et al., BSSA 2010: Eq.(8):
    # constants:
    dd = dip  * np.pi/180.0
    rr = rake * np.pi/180.0
    # change of the mean compressive (--> negative sign) stress:
    dsigma = - (sxx + syy + szz)/3.0
    # pore pressure:
    dp = skempton * dsigma
    dsig = - ( syy * np.square(np.sin(dd)) + szz * np.square(np.cos(dd)) - syz * np.sin(2.0*dd) )  # compressive --> negative sign
    dtau = ( sxy * np.sin(dd) - szx * np.cos(dd) ) * np.cos(rr) - ( 0.5 * np.sin(2.0*dd) * (syy - szz) + syz * (np.square(np.sin(dd)) - np.square(np.cos(dd))) ) * np.sin(rr)
    dcfs = dtau - friction * ( dsig  - dp ) 
    return dcfs, dp, dtau, dsig


def greensfunction_convolution(z, ti, dti, wi, Tyear, D, r):
    # constants:
    factor = z/(8.0*np.power(np.pi,1.5)*np.power(D,2.5))
    d_t = dti * year2sec
    tw = (Tyear - ti[(ti<Tyear)]) * year2sec
    G = np.sum(wi[(ti<Tyear)] * (np.exp(- np.square(r)/(4.0*D*tw)) / np.power(tw,2.5))) * D * factor  * d_t
    return G

def CFS_calculation(Asig, ta, lat, lon, sourcelat, sourcelon, sourcearea, D, strike, dip, rake, friction, skempton, zkm, ti0, w_height, T2, T3):
    dottau = 1e6 * Asig / ta
    z = 1000.0 * zkm                 # [m]
    density = 1000.0                 # density of water [kg/m^3] 
    gravity = 9.81                   # [m/s^2]
    fac = sourcearea * density * gravity    # [N/m] pressure/height, directed downwards
    dti = ti0[1] - ti0[0]
    # distance to source [m]:
    rxy = 1000.0 * dist(lat, lon, sourcelat, sourcelon)
    r = np.sqrt( np.square(rxy) + np.square(z) )
    # effect of static loading (Deng etal. BSSA 2010 Eqs.6-8):
    sxx, syy, szz, sxy, szx, syz = dengStresstensor(fac,strike,lat,lon,z,sourcelat,sourcelon)
    for nt in range(2):
        if nt == 0:
            T = T2
        elif nt == 1:
            T = T3
        ti = leveltime[(leveltime<=T)]
        w_h = w_height[:, (leveltime<=T)]
        sxxi = np.sum(sxx * w_h[:,-1])
        syyi = np.sum(syy * w_h[:,-1])
        szzi = np.sum(szz * w_h[:,-1])
        sxyi = np.sum(sxy * w_h[:,-1])
        szxi = np.sum(szx * w_h[:,-1])
        syzi = np.sum(syz * w_h[:,-1])
        # static loading:
        CFSs, dp, dtau, dsig = dengCFS(dip,rake,friction,skempton,sxxi,syyi,szzi,sxyi,szxi,syzi)
        pdiff = 0.0
        for ii in range(len(sourcelat)):
            pdiff += fac * greensfunction_convolution(z, ti, dti, w_h[ii,:], ti0[i], D, r[ii])
        CFSi = (dottau*(T-ti0[0]) + CFSs + friction * pdiff) / 1e6     # [MPa]
        if nt == 0:
            CFS2 = CFSi
        elif nt == 1:
            CFS3 = CFSi
    return CFS2, CFS3

############################ Paths to Directories #############################
Reservoir_path = '../Data/Lake/Reservoir_sourcepoints.dat'  ## Directory of reservoir source points
Water_level_path = '../Data/Lake/Gotvand_water_level.txt'   ## Directory of water level of Gotvand Dam
CFS_dir = '../OutPut/CFS'                                   ## Directory for saving stress results               

########################## Definition of Parameters ###########################
ncase = 0

if ncase == 0:
    dip = 60.0
    Asig = 0.01          # [MPa]
elif ncase == 1:
    dip = 25.0
    Asig = 0.01          # [MPa]
elif ncase == 2:
    dip = 60.0
    Asig = 0.03          # [MPa]
elif ncase == 3:
    dip = 25.0
    Asig = 0.03          # [MPa]

D = 1.0              # [m2/s]
depth = 12.5         # [km]

dottau = 0.002       # [MPa/yr]
ta = Asig / dottau   # [year]

strike = 300.0
rake = 85.0

# T1: 30.07.2011
T1 = float(timeyear([2011], [7], [30], [12], [0], [0]))
# T2: 09.09.2014
T2 = float(timeyear([2014], [9], [9], [12], [0], [0]))
# T3: 03.03.2019 (Flood)
T3 = float(timeyear([2019], [2], [28], [12], [0], [0]))

friction = 0.8
skempton = 0.5
poisson = 0.3

# Define observation grid:
dx = 0.02
lat1 = 31.6
lat2 = 32.8
lon1 = 48.6
lon2 = 49.9
lat = np.arange(lat1, lat2+0.5*dx, dx) # (31.60, 32.8, 0.02)
lon = np.arange(lon1, lon2+0.5*dx, dx) # (48.60, 49.90, 0.02)
        
########################### Reading the Data ################################## 
### reservoir source points ===================================================
cell_dimention = 0.005  # [degree]
data = np.loadtxt(Reservoir_path, skiprows=2)
sourcelat = data[:,2]
sourcelon = data[:,1]
topo_level = data[:,3]

## calculating area of source point
# Using haversine formula to convert distance in degree to meter
'''
One degree of latitude is approximately 111 kilometers (111,000 meters) but 
One degree of longitude varies based on the latitude. We consider the center of 
our study area as reference latitude to calculate the length of one degree 
of longitude at the given latitude.
'''
ref_lat = (lat1 + lat2)/2
meters_per_degree_lon = 111000 * math.cos(math.radians(ref_lat))
sourcearea = cell_dimention * 111000 * cell_dimention * meters_per_degree_lon # [m^2]

### water level ===============================================================
data = np.loadtxt(Water_level_path, skiprows=1)
NT = len(data[:,2])
ty = timeyear(data[:,0], data[:,1], data[:,2], 12*np.ones(NT), np.zeros(NT), np.zeros(NT))
ll = data[:, 3]  
ind = (ty<=T3)
leveltime = ty[ind]  # time of water level value
level = ll[ind]      # water level [m]

### to avoid initial load step:
topo_level[(topo_level<level[0])] = level[0]
w_height = np.zeros((len(topo_level), len(level)))
for i in range(len(topo_level)):
    w_height[i,:] = level - topo_level[i]
    w_height[i, (w_height[i,:] < 0)] = 0.0

############################ Calculations ##############################
CFS2 = np.zeros((len(lat), len(lon)))  # [MPa] CFS at T2
CFS3 = np.zeros((len(lat), len(lon)))  # [MPa] CFS at T3
n = 0
count = 0
for i in range(len(lat)):
    for j in range(len(lon)):
        CFS2[i, j], CFS3[i, j] = CFS_calculation(Asig, ta, lat[i], lon[j], sourcelat, sourcelon, sourcearea, D, strike, dip, rake, friction, skempton, depth, leveltime, w_height, T2, T3)
        if np.floor(n/100.0) > count:
            sys.stdout.write('\r'+str('\t ncase=%d n=%d/%d: Rr1=%f  Rr2=%f\r' % (ncase, n, len(lat)*len(lon), CFS2[i,j], CFS3[i,j])))
            sys.stdout.flush()
            count += 1
        n += 1

######################### Saving Seismicity Density ###########################        
suff = 'Asig%.3fMPa_dottau%.3fMPa_strike%.1f_dip%.1f_rake%.1f_z%.1fkm_D%.2fm2s' % (Asig, dottau, strike, dip, rake, depth, D)

outname1 = '%s/CFS_%s_impoundment.out' % (CFS_dir, suff)
np.savetxt(outname1, CFS2)
print('\n\t OUTPUT: %s' % (outname1))

outname2 = '%s/CFS_%s_post.out' % (CFS_dir, suff)
np.savetxt(outname2, CFS3)
print('\t OUTPUT: %s\n' % (outname2))


  



