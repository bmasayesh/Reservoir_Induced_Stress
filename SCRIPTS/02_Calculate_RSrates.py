# GFZ & Uni Potsdam
# Date: November 2024
# Authors: Behnam Maleki Asayesh & Sebastian Hainzl
'''
This code calculate seismicity rate using Coulomb-Rate-and-State (RS) model.
'''

########################## Importing Required Modules #########################
import sys
import math
import datetime as dt
from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# from matplotlib.colorbar import ColorbarBase
# from matplotlib.colors import Normalize
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

# Function for Calculation of Distance in 2D ==================================
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

# Function for calculation of stress matrix based on Deng et al., BSSA 2010 ===
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

# Function for calculation of Coulomb stress changes ==========================         
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

def RSrate(ti, Si, r0, Asig, ta):
    '''
    Seismicity response of RS for a continuous stress evolution S(t) for times t
    according to Heimisson & Segall, JGR 2018, Eq.(20) & Eq.(29) & Eq.(34)
    '''
    dt = np.ediff1d(ti, to_end=ti[-1]-ti[-2])
    K = np.exp(Si/Asig)
    integK = np.cumsum(K * dt)
    R = r0 * K / (1.0 + integK / ta)
    return R

def Rate_calculation(Asig, ta, lat, lon, sourcelat, sourcelon, sourcearea, D, strike, dip, rake, friction, skempton, zkm, tw, w_height):
    dottau = 1e6 * Asig / ta
    z = 1000.0 * zkm                 # [m]
    density = 1000.0                 # density of water [kg/m^3] 
    gravity = 9.81                   # [m/s^2]
    fac = sourcearea * density * gravity    # [N/m] pressure/height, directed downwards
    # distance to source [m]:
    rxy = 1000.0 * dist(lat, lon, sourcelat, sourcelon)
    r = np.sqrt( np.square(rxy) + np.square(z) )
    # effect of static loading (Deng etal. BSSA 2010 Eqs.6-8):
    sxx, syy, szz, sxy, szx, syz = dengStresstensor(fac,strike,lat,lon,z,sourcelat,sourcelon)
    pgreens = np.zeros((len(r), len(tw)-1))
    dtw = tw[1:] - tw[0]
    NA = np.newaxis
    for ii in range(len(sourcelat)):
        pgreens += np.exp(-np.square(r[:,NA]) / (4.0 * D * dtw[NA,:])) / np.power(dtw[NA,:], 2.5)
    dti = (tw[1] - tw[0]) * year2sec
    factor = z / (8.0 * np.power(np.pi, 1.5) * np.power(D, 2.5))
    pgreens *= D * factor * dti
    # CFS calculation:
    CFS = np.zeros(len(tw))
    for i in range(len(tw)):
        w_h0 = w_height[:, i]
        sxxi = np.sum(sxx * w_h0)
        syyi = np.sum(syy * w_h0)
        szzi = np.sum(szz * w_h0)
        sxyi = np.sum(sxy * w_h0)
        szxi = np.sum(szx * w_h0)
        syzi = np.sum(syz * w_h0)
        # static loading:
        CFSs, dp, dtau, dsig = dengCFS(dip,rake,friction,skempton,sxxi,syyi,szzi,sxyi,szxi,syzi)
        pdiff = np.sum(w_height[:, :i] * np.flip(pgreens[:, :i], axis=1))
        CFS[i] = (dottau*(tw[i]-tw[0]) + CFSs + friction * pdiff) / 1e6     # [MPa]
    Rr0 = RSrate(tw, CFS, 1.0, Asig, ta)
    return Rr0

############################ Paths to Directories #############################
Reservoir_path = '../Data/Lake/Reservoir_sourcepoints.dat'  ## Directory of reservoir source points
Water_level_path = '../Data/Lake/Gotvand_water_level.txt'   ## Directory of water level of Gotvand Dam
RS_dir = '../OutPut/RS'                                     ## Directory for saving seismicity rate               

########################### Declaring of Parameters ###########################
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
# Flood: 3.3.2019
T3 = float(timeyear([2019], [3], [1], [12], [0], [0]))

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

meanRr1 = np.zeros((len(lat), len(lon)))  # R/r integrated in [T1, T2] / T
meanRr2 = np.zeros((len(lat), len(lon)))  # R/r integrated in [T2, T3] / T
n = 0
for i in range(len(lat)):
    for j in range(len(lon)):

        Rr = Rate_calculation(Asig, ta, lat[i], lon[j], sourcelat, sourcelon, sourcearea, D, strike, dip, rake, friction, skempton, depth, leveltime, w_height)

        ind = ((leveltime>=T1) & (leveltime<T2))
        meanRr1[i,j] = np.trapz(Rr[ind], leveltime[ind]) / (T2 - T1)
        ind = ((leveltime>=T2) & (leveltime<T3))
        meanRr2[i,j] =  np.trapz(Rr[ind], leveltime[ind]) / (T3 - T2)
        #print('\t ncase=%d n=%d/%d: Rr1=%f  Rr2=%f' % (ncase, n, len(lat)*len(lon), meanRr1[i,j], meanRr2[i,j]))
        sys.stdout.write('\r'+str('\t ncase=%d n=%d/%d: Rr1=%f  Rr2=%f\r' % (ncase, n, len(lat)*len(lon), meanRr1[i,j], meanRr2[i,j])))
        sys.stdout.flush()
        n += 1

######################### Saving Seismicity Density ###########################        
suff = 'Asig%.3fMPa_dottau%.3fMPa_strike%.1f_dip%.1f_rake%.1f_z%.1fkm_D%.2fm2s' % (Asig, dottau, strike, dip, rake, depth, D)

outname1 = '%s/meanRr_%s_impoundment.out' % (RS_dir, suff)
np.savetxt(outname1, meanRr1)
print('\n\t OUTPUT: %s' % (outname1))

outname2 = '%s/meanRr_%s_post.out' % (RS_dir, suff)
np.savetxt(outname2, meanRr2)
print('\t OUTPUT: %s\n' % (outname2))


  



