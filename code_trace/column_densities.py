import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from astropy import constants, units
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import healpy as hp
import plons
import plons.ConversionFactors_cgs as cgs
import magritte.setup as setup
import magritte.core as magritte
import magritte.tools as tools
from tqdm import tqdm
import numpy as np
import utils_raytracer as rtf
import cons
import time 
import os

########################################################################################################################

# Read in the model 

########################################################################################################################

timer_start = time.time()

# Allow dump number to be set via environment variable (for batch processing)
# Otherwise use default values
dump = int(os.environ.get('DUMP_NUM', 40))

wdir = Path.cwd()

# Working directory where phantomanalysis runs (contains trace_output/ and PhotoData/)
work_dir = Path(os.environ.get('WORK_DIR', '/fred/oz304/beckers/pigru'))
av_output_dir = work_dir / 'AV'
av_output_dir.mkdir(exist_ok=True)

# Data directory and file prefix (e.g. 'pigru' or 'wind')
data_dir = Path(os.environ.get('DATA_DIR', '/fred/oz304/tdanilov/pigru'))
prefix   = os.environ.get('PREFIX', 'pigru')

Output = "PhotoData" 

# Depending on where you put the data, you might need to change the paths here
dump_file  = data_dir / f'{prefix}_{dump:05d}'
print(dump_file)
setup_file = data_dir / f'{prefix}.setup'
print(setup_file)
input_file = data_dir / f'{prefix}.in'

# Loading the data
setupData = plons.LoadSetup(data_dir.parent, f'{data_dir.name}/{prefix}')
dumpData  = plons.LoadFullDump(str(dump_file), setupData)
print(dumpData.keys())

code_time = dumpData._params['time'] 
unit_time = dumpData._params['utime']
real_time = code_time * unit_time  # in seconds
print(f"Real simulation time: {real_time} seconds")
position = np.stack((dumpData['x'], dumpData['y'], dumpData['z']), axis=-1)*1e-2

# position = dumpData['position']*1e-2  
velocity = np.stack((dumpData['vx'], dumpData['vy'], dumpData['vz']), axis=-1)*1e3
velocity = velocity/constants.c.si.value # velocity vectors        [m/s  -> 1/c]

rho      = np.array(dumpData["rho"] )              # density                 [g/cm^3]
u        = np.array(dumpData["u"]   )              # internal energy density [erg/g]
tmp      = np.array(dumpData["temp"])              # temperature             [K]

r = (np.linalg.norm(position, axis=1)*units.m ).to(units.au)    # Spherical radius

npoints = len(rho)

# Convenience arrays
zeros = np.zeros(npoints)
ones  = np.ones (npoints)

# Define turbulence at 1 km/s, doesnt do anything, needed for Magritte modelo
trb = (1000.0/constants.c.si.value)**2 * ones

# Line data file, needed to create the Magritte model (doesnt do anything in this case)
lamda_file = Path('test.txt')   

# conv = (1.0*units.m).to(units.au).value
timer_reader = time.time()

print('')
print(f"Time to read in data: {timer_reader - timer_start} seconds")
print('')

########################################################################################################################

# Create the Delauney mesh and find the boundary particles + save to file for later just in case

########################################################################################################################

Output = work_dir / 'PhotoData'
Output.mkdir(exist_ok=True)

target = Output / f'boundary{dump}.txt'
target2 = Output / f'neighbors{dump}.pkl'

if target.exists() and target2.exists():
    print(f'File {target} and {target2} exists, skipping computation.')
    with open(Output / f'neighbors{dump}.pkl', 'rb') as f:
        neighbors = pickle.load(f)
        
    boundary = np.loadtxt(Output / f'boundary{dump}.txt', dtype=int)
    nbs       = [n for sublist in neighbors for n in sublist]
    n_nbs     = [len(sublist) for sublist in neighbors]
    
else:
    print(f'Computing and saving to {target}.')
    delaunay = Delaunay(position, qhull_options='QJ')
    indptr, indices = delaunay.vertex_neighbor_vertices
    neighbors = [indices[indptr[k]:indptr[k+1]] for k in range(npoints)]
    
    nbs       = [n for sublist in neighbors for n in sublist]
    n_nbs     = [len(sublist) for sublist in neighbors]

    mask = delaunay.neighbors == -1
    boundary_simplices = delaunay.simplices[mask.any(axis=1)]
    boundary = np.unique(boundary_simplices.ravel())

    b_nms = np.linalg.norm(position[boundary], axis=1)
    p_nms = np.linalg.norm(position, axis=1)
    boundary = np.where(p_nms >= b_nms.min())[0]

    np.savetxt(Output / f'boundary{dump}.txt', boundary, fmt='%d')
    with open(Output / f'neighbors{dump}.pkl', 'wb') as f:
        pickle.dump(neighbors, f)
        
timer_delaunay = time.time()

print('')
print(f"Time to calculate the delauney mesh: {timer_delaunay - timer_reader} seconds")
print('')


########################################################################################################################

# Calculate H2 Density

########################################################################################################################

# Physical constants (CGS units)
kboltz = 1.380649e-16              # Boltzmann constant (erg/K)
patm = 1.01325e6                   # Atmospheric pressure (dyn/cm²)
atomic_mass_unit = 1.660538921e-24 # Atomic mass unit (g)
mass_proton_cgs = 1.6726219e-24    # Proton mass (g)

# Get elemental abundances and molecular coefficients
eps = cons.get_eps()
coefficients = cons.get_coefficients()
molecules = np.array(list(coefficients.keys()))

# Atomic weights: H, He, C, O, N, Ne, Si, S, Fe, Ti
Aw_nElements = [1.0079, 4.0026, 12.011, 15.9994, 14.0067, 
                20.17, 28.0855, 32.06, 55.847, 47.867]

# Calculate mean molecular mass per hydrogen atom
mass_per_H = atomic_mass_unit * np.dot(Aw_nElements, np.array(list(eps.values())))
# print(f"Mass per H atom: {mass_per_H} g")

# Total hydrogen partial pressure
pH_tot = rho * kboltz * tmp / (patm * mass_per_H)

# Solve quadratic equation for H2 dissociation: a*pH^2 + b*pH + c = 0
a = 2 * cons.calc_Kd(coefficients['H2'], tmp)
a[a == 0] = 1e-30
b = 1.0
c = -pH_tot

pH = cons.solve_q(a, b, c)
pH2 = pH**2 * cons.calc_Kd(coefficients['H2'], tmp)

# Convert to number densities with proper units
patm_u = patm * units.dyn / units.cm**2
kboltz_u = kboltz * units.erg / units.K
T_u = tmp * units.K
rho_u = rho * units.g / units.cm**3

pH2_u = pH2 * units.erg / (units.cm * units.dyn)
pH_u = pH * units.erg / (units.cm * units.dyn)

# Calculate number densities (particles per cm^3)
nH2_p = (pH2_u * patm_u / (kboltz_u * T_u)).to(units.cm**-3)
nH_p = (pH_u * patm_u / (kboltz_u * T_u)).to(units.cm**-3)

# Final outputs in SI units (particles per m^3)
nH2 = nH2_p.to(units.m**-3).value
nH = nH_p.to(units.m**-3).value

########################################################################################################################

# Trace an amount of rays which we will be using for the interpolation 

########################################################################################################################

# Define the resolution of the HEALPix map, these rays will be used for the interpolation
nside_interp = 2**3  
npix_interp = hp.nside2npix(nside_interp) 

# Get the angles (theta, phi) for each pixel center
theta_interp, phi_interp = hp.pix2ang(nside_interp, np.arange(npix_interp))

x_interp = np.sin(theta_interp) * np.cos(phi_interp)
y_interp = np.sin(theta_interp) * np.sin(phi_interp)
z_interp = np.cos(theta_interp)

unit_vectors_interp = np.vstack((x_interp, y_interp, z_interp)).T  # Shape: (npix, 3)

print(f"Tracing {npix_interp} rays")

#Define the starting positions of the rays on the surface of the AGB star
posAGB = ((dumpData._params['posAGB'])*units.cm).to(units.m)
R_star = (setupData['primary_Reff'])*units.au

R_x = (R_star * np.sin(theta_interp) * np.cos(phi_interp) ) . to(units.m)
R_y = (R_star * np.sin(theta_interp) * np.sin(phi_interp) ) . to(units.m)
R_z = (R_star * np.cos(theta_interp) ) . to(units.m)

start_points = np.vstack((R_x, R_y, R_z)).T
start_points = (start_points + posAGB).value

#Test it out for one ray (also compiles the njit for speedup, if you run this a second time it should be instant)
index = 0
ray_points, ray_indices = rtf.get_all_points(start_points[index], unit_vectors_interp[index], neighbors, position, boundary)

def process_ray(start_point, ray):
        ray_points, ray_indices = rtf.get_all_points(start_point, ray, neighbors, position, boundary)
        return ray_indices, ray_points

rays = unit_vectors_interp

# Pre-allocate arrays 
# Adjust max_points to match your expected ray length (500 should be fine)
max_points = 500
num_rays = len(rays)

all_indices = np.zeros((num_rays, max_points), dtype=int)
all_positions = np.zeros((num_rays, max_points, 3))
all_r = np.zeros((num_rays, max_points))

# change max_workers based on your CPU cores (if running on mac, use ThreadPoolExecutor)
with ProcessPoolExecutor(max_workers=16) as executor:
# with ThreadPoolExecutor(max_workers=12) as executor:
    futures = {executor.submit(process_ray, start_points[i], ray): i 
            for i, ray in enumerate(rays)}
    
    for future in tqdm(as_completed(futures), total=len(futures), 
                    desc='Rays processed', miniters=1):
        idxx = futures[future]
        ray_indices, ray_points = future.result()
        
        # Get actual length of this ray's data
        actual_length = len(ray_indices)
        
        # Fill arrays with data (up to actual_length)
        all_indices[idxx, :actual_length] = ray_indices
        all_positions[idxx, :actual_length] = ray_points
        all_r[idxx, :actual_length] = np.linalg.norm(ray_points, axis=1)
        
print("All rays processed.")

indices_for_Magritte = all_indices.flatten()[all_indices.flatten() != 0]
print(f"Total unique indices for Magritte: {len(np.unique(indices_for_Magritte))}")

timer_rays = time.time()

print('')
print(f"Time to trace rays: {timer_rays - timer_delaunay} seconds")
print('')

########################################################################################################################

# Create Magritte model file

########################################################################################################################

models_dir = work_dir / 'Models'
if not models_dir.exists():
    models_dir.mkdir()

model_file = models_dir / f'Model{dump}.hdf5'

if model_file.exists():
    model_file.unlink()

model = magritte.Model ()                              # Create model object

model.parameters.set_model_name         (str(model_file))   # Magritte model file
model.parameters.set_dimension          (3)            # This is a 3D model
model.parameters.set_npoints            (npoints)      # Number of points

nrays = 192
model.parameters.set_nrays              (nrays)            # Number of rays 

model.parameters.set_nspecs             (3)            # Number of species
model.parameters.set_nlspecs            (1)            # Number of line species
model.parameters.set_nquads             (1)           # Number of quadrature points

model.geometry.points.position.set(position)
model.geometry.points.velocity.set(velocity)

model.geometry.points.  neighbors.set(  nbs)
model.geometry.points.n_neighbors.set(n_nbs)

model.chemistry.species.abundance = np.array((zeros, nH2, zeros)).T
model.chemistry.species.symbol    = ['test', 'H2', 'e-']

model.thermodynamics.temperature.gas  .set(tmp)
model.thermodynamics.turbulence.vturb2.set(trb)

model.parameters.set_nboundary(boundary.shape[0])
model.geometry.boundary.boundary2point.set(boundary)

model = setup.set_uniform_rays            (model)   # Uncomment to use all directions
model = setup.set_boundary_condition_CMB  (model)
#model = setup.set_boundary_condition_zero  (model)

transitions = [0]
model = setup.set_linedata_from_LAMDA_file(model, lamda_file, {'considered transitions': transitions})
model = setup.set_quadrature              (model)

model.write()

timer_magritte = time.time()

print('')
print(f"Time to create Magritte model: {timer_magritte - timer_rays} seconds")
print('')

########################################################################################################################

# Calculate the column densities

########################################################################################################################

model = magritte.Model(str(model_file))
model.density.set(nH2)

model.set_column_points(indices_for_Magritte)

timer_cd = time.time()

print('')
print(f"Time to calculate column densities points: {timer_cd - timer_magritte} seconds")
print('')

# Calculate the solid angle averaged column densities
nside = hp.npix2nside(nrays) 
dOmega = hp.nside2pixarea(nside)
cd_nH2_dOmega = (((np.array(model.column)*units.m**(-2)).to(units.cm**(-2)) ).value).T * dOmega
cda = np.sum(cd_nH2_dOmega, axis=1) * 1/ (4*np.pi)  #calculates sum(NH_2 * dOmega) / 4pi
a_xs = cda[all_indices]

#Interpolate on the nearest rays
theta, phi = hp.vec2ang(position)
all_ngb_i = hp.get_all_neighbours(nside_interp, theta, phi).T
ngb = np.array([unit_vectors_interp[i] for i in all_ngb_i])

#expand position to the same shape as ngb
pos_new = np.repeat(position[:, np.newaxis, :], len(ngb[1]), axis = 1)
Distances = np.linalg.norm(np.cross(pos_new, ngb), axis=2) / np.linalg.norm(ngb, axis=2)

#keep only the eight closest rays
# Find indices of the n_nbs closest rays for each position
n_nbs = 8
r_r = ((np.linalg.norm(position , axis = 1)*units.m).to(units.cm)).value
r_r_new = np.repeat(r_r[:, np.newaxis], n_nbs, axis = 1)/100

closest_indices = np.argsort(Distances, axis=1)[:, :n_nbs]  
# Select the closest rays and their distances
closest_rays = np.take_along_axis(ngb, closest_indices[:, :, np.newaxis], axis=1) #Not used but whatever
closest_rays_indices = np.take_along_axis(all_ngb_i, closest_indices, axis=1)  
new_distances = np.take_along_axis(Distances, closest_indices, axis=1)

res = rtf.do_interpolation(r_r_new, closest_rays_indices, a_xs, all_r)
x_vals = rtf.do_calculate_x(res, new_distances, 2)
dumpData['x_vals'] = x_vals

# Get particle IDs
particle_ids = np.array(dumpData['iorig'], dtype=np.int32)

# Convert column density to A_V
A_V = 2*x_vals / (1.87e21)  # Visual extinction [mag]

# Save all particles as a Fortran-readable flat binary (no extension, like phantom dumps):
# int32: number of particles | int32[n]: particle IDs (iorig) | float64[n]: A_V values
av_file = av_output_dir / f'AV_{dump:05d}'
n = len(particle_ids)
with open(av_file, 'wb') as f:
    np.array([n], dtype=np.int32).tofile(f)
    particle_ids.tofile(f)
    A_V.astype(np.float64).tofile(f)
print(f'Saved A_V binary for {n} particles to {av_file}')

timer_end = time.time()

print('')
print(f"Total runtime: {timer_end - timer_start} seconds")

# if you want to make plots of the column density slices uncomment the following

# n = 600
# lims = 5000  

# import plons
# import plons.SmoothingKernelScript    as sk
# import plons.PhysicalQuantities       as pq
# import plons.ConversionFactors_cgs    as cgs
# import plons.Plotting                 as plot

# import numpy.typing as npt
# import matplotlib
# from typing import Dict, Tuple, Any, Optional

# def plotSlice(ax: plt.Axes,
#             X: npt.NDArray[np.single],
#             Y: npt.NDArray[np.single],
#             smooth: Dict[str, npt.NDArray[np.single]],
#             observable: str,
#             logplot: bool = False,
#             fs : int = 16,
#             cbar : bool = False,
#             cmap: matplotlib.colors.Colormap = plt.cm.get_cmap('inferno'),
#             clim: Tuple[Optional[float], Optional[float]] = (None, None)) -> matplotlib.colorbar.Colorbar:
#     """Plot a property given a grid and smoothed data ontop of the grid

#     Args:
#         ax (plt.Axes): axis of figure on which you want to plot the slice
#         X (npt.NDArray[np.single]): X values in meshgrid which you want to plot
#         Y (npt.NDArray[np.single]): Y values in meshgrid which you want to plot
#         smooth (Dict[str, npt.NDArray[np.single]]): Dictionary pointing at smoothed values in meshgrid which you want to plot
#         observable (str): Name of the observable you want to plot, corresponding to the name in the smooth directory
#         logplot (bool, optional): plot in log scale?. Defaults to False.
#         cmap (matplotlib.colors.Colormap, optional): Colormap to use. Defaults to cm.get_cmap('inferno').
#         clim (Tuple[Optional[float], Optional[float]], optional): limits for the colorbar. Defaults to (None, None).

#     Returns:
#         colorbar.Colorbar: Colorbar
#     """

#     ax.set_aspect('equal')
#     ax.set_facecolor('k')

#     if logplot:
#         obs = np.log10(smooth[observable]+1e-99)
#     else:
#         obs = smooth[observable]
#     axPlot = ax.pcolormesh(X/cgs.au, Y/cgs.au, obs, cmap=cmap, vmin=clim[0], vmax = clim[1])
    
#     if cbar == True:
#         cbar = plt.colorbar(axPlot, ax = ax, location='right', fraction=0.0471, pad=0.01)  
#         cbar.ax.tick_params(labelsize=fs-4)          
#         return cbar
    
# def get_smooth(dumpData, X, Y, Z, key):
#     smooth = sk.smoothMesh(X, Y, Z, dumpData, [f'{key}'])
#     return remove_nans(smooth)

# def remove_nans(smooth):
#     for key in smooth.keys():
#         smooth[key][np.isnan(smooth[key])] = 0
#     return smooth

# x = np.linspace(-lims, lims, n)*cgs.au
# y = np.linspace(-lims, lims, n)*cgs.au
# X, Y = np.meshgrid(x, y)
# Z = np.zeros_like(X)

# N_z = get_smooth(dumpData, X, Y, Z, 'x_vals')

# fig, ax = plt.subplots(1, 1, figsize=(8,6))

# cbar = plotSlice(ax, X, Y, N_z, f'x_vals', logplot = True, cmap = plt.colormaps['inferno'], cbar = True, 
#         clim=(np.log10( min(np.array(dumpData['x_vals'][ np.array(dumpData['x_vals']) > 0 ])) ), np.log10( np.amax(N_z['x_vals']) ) ) )

# cbar.set_label('log N [cm$^{-2}$]', fontsize=13)

# ax.set_xlabel('x [au]', fontsize=13)
# ax.set_ylabel('y [au]', fontsize=13)

# fig.savefig(f'ColumnDensity_xy_{dump}_nrays{nrays}_nside{nside_interp}.png', dpi=300)
# plt.close()

# # Plot density projections (x-z plane)
# x = np.linspace(-lims, lims, n)*cgs.au
# z = np.linspace(-lims, lims, n)*cgs.au
# X, Z = np.meshgrid(x, z)
# Y = np.zeros_like(X)

# N_x = get_smooth(dumpData, X, Y, Z, 'x_vals')

# fig, ax = plt.subplots(1, 1, figsize=(8,6))

# cbar = plotSlice(ax, X, Z, N_x, f'x_vals', logplot = True, cmap = plt.colormaps['inferno'], cbar = True, 
#         clim=(np.log10( 4.163480925788e+16 ), np.log10( np.amax(N_x['x_vals']) ) ) )

# cbar.set_label('log N [cm$^{-2}$]', fontsize=13)

# ax.set_xlabel('x [au]', fontsize=13)
# ax.set_ylabel('z [au]', fontsize=13)

# fig.savefig(f'ColumnDensity_xz_{dump}_nrays{nrays}_nside{nside_interp}.png', dpi=300)
# plt.close()

# # Plot density projections (x-y plane)
# x = np.linspace(-lims, lims, n)*cgs.au
# y = np.linspace(-lims, lims, n)*cgs.au
# X, Y = np.meshgrid(x, y)
# Z = np.zeros_like(X)

# rho_z = get_smooth(dumpData, X, Y, Z, 'rho')

# fig, ax = plt.subplots(1, 1, figsize=(8,6))

# cbar = plotSlice(ax, X, Y, rho_z, 'rho', logplot=True, cmap=plt.colormaps['inferno'], cbar=True,
#         clim=(np.log10(min(np.array(dumpData['rho'][np.array(dumpData['rho']) > 0]))), 
#               np.log10(np.amax(rho_z['rho']))))

# cbar.set_label('log $\\rho$ [g / cm$^{3}$]', fontsize=13)

# ax.set_xlabel('x [au]', fontsize=13)
# ax.set_ylabel('y [au]', fontsize=13)

# fig.savefig(f'Density_xy_{dump}.png', dpi=300)
# plt.close()

# # Plot density projections (x-z plane)
# x = np.linspace(-lims, lims, n)*cgs.au
# z = np.linspace(-lims, lims, n)*cgs.au
# X, Z = np.meshgrid(x, z)
# Y = np.zeros_like(X)

# rho_x = get_smooth(dumpData, X, Y, Z, 'rho')

# fig, ax = plt.subplots(1, 1, figsize=(8,6))

# cbar = plotSlice(ax, X, Z, rho_x, 'rho', logplot=True, cmap=plt.colormaps['inferno'], cbar=True,
#         clim=(np.log10(min(np.array(dumpData['rho'][np.array(dumpData['rho']) > 0]))), 
#               np.log10(np.amax(rho_x['rho']))))

# cbar.set_label('log $\\rho$ [g / cm$^{3}$]', fontsize=13)

# ax.set_xlabel('x [au]', fontsize=13)
# ax.set_ylabel('z [au]', fontsize=13)

# fig.savefig(f'Density_xz_{dump}.png', dpi=300)
# plt.close()

