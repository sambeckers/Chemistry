import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

savedirmain = Path('/Users/sam/Documents/GitHub/Chemistry')
mf = 'evolving_model'
of = 'ev_output'

out = np.genfromtxt(savedirmain / mf / of / "ev_13775.dat", comments="#", skip_header=4, names=True)
t = out['TIME']

fig, axes = plt.subplots(4, 1, figsize=(10, 12), dpi=300, sharex=True)
# rho
axes[0].plot(t, out['DENSITY'], color='k', lw=2)
axes[0].set_ylabel('$\\rho$ [cm$^{-3}$]')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.7)
# T
axes[1].plot(t, out['TEMP'], color='r', lw=2)
axes[1].set_ylabel('$T$ [K]')
axes[1].set_yscale('log')
axes[1].grid(True, alpha=0.7)
# A_V
axes[2].plot(t, out['AV'], color='b', lw=2)
axes[2].set_ylabel('$A_V$ [mag]')   
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.7)
# CO abundance
axes[3].plot(t, out['CO'], color='magenta', lw=2)
axes[3].set_ylabel('CO Abundance (wrt H$_{nuc}$)')
axes[3].set_xlabel('Time [yr]')
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].grid(True, alpha=0.7)

plt.tight_layout()
# plt.savefig(savedirmain / 'figures' / f'evolution_params5001.pdf', bbox_inches='tight', dpi=300)
plt.show()

fig = plt.figure(dpi=300)
ax = plt.axes(projection='3d') # 3D projection
scatter = ax.scatter(out['X'], out['Y'], out['Z'], c=out['TIME'], cmap='inferno', s=20) # Color by TIME
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('Time [yr]', labelpad=15 )
plt.tight_layout()
plt.title("3D trace")
# plt.savefig(savedirmain / 'figures' / f'evolution_3Dtrace5001.pdf', bbox_inches='tight', dpi=300)
plt.show()
