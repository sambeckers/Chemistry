import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
# Set plt font to LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica"
})
from pathlib import Path
from tqdm import tqdm

def load_all_particles(particle_IDs):
    """Load all particle trace data and compute global bounds."""
    all_data = {}
    for pid in particle_IDs:
        filepath = savedirmain / mf / of / f"ev_{pid}.dat"
        data = np.genfromtxt(filepath, comments="#", skip_header=4, names=True)
        all_data[pid] = data

    # Collect all data across all particles
    all_x = np.concatenate([data['X'] for data in all_data.values()])
    all_y = np.concatenate([data['Y'] for data in all_data.values()])
    all_z = np.concatenate([data['Z'] for data in all_data.values()])
    all_times = np.concatenate([data['TIME'] for data in all_data.values()])
    all_density = np.concatenate([data['DENSITY'] for data in all_data.values()])
    all_temp = np.concatenate([data['TEMP'] for data in all_data.values()])
    all_av = np.concatenate([data['AV'] for data in all_data.values()])
    all_r = np.sqrt(all_x**2 + all_y**2 + all_z**2)
    
    bounds = {
        'x': (all_x.min(), all_x.max()),
        'y': (all_y.min(), all_y.max()),
        'z': (all_z.min(), all_z.max()),
        'r': (all_r.min(), all_r.max()),
        'time': np.sort(np.unique(all_times)),
        'density': (all_density.min(), all_density.max()),
        'temp': (all_temp.min(), all_temp.max()),
        'av': (all_av.min(), all_av.max())
    }
    
    return all_data, bounds

def plot_all_params_time(all_data, bounds, particle_IDs, save=True):
    """Plot density, temperature, and A_V for all particles."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), dpi=300, sharex=True)
    
    # Generate colors for all particles
    colors = plt.cm.Oranges(np.linspace(0.2, 1, len(particle_IDs)))
    
    for i, pid in enumerate(particle_IDs):
        data = all_data[pid]
        t = data['TIME']

        # Density
        axes[0].plot(t, data['DENSITY'], color=colors[i], lw=2, 
                    label=f'{pid}')
        
        # Temperature
        axes[1].plot(t, data['TEMP'], color=colors[i], lw=2, 
                    label=f'{pid}')

        # A_V
        axes[2].plot(t, data['AV'], color=colors[i], lw=2, 
                    label=f'{pid}')

    # Format axes
    axes[0].set_ylabel('$\\rho$ [cm$^{-3}$]', fontsize=16)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(bounds['time'].min(), bounds['time'].max())
    axes[0].set_ylim(bounds['density'])
    axes[0].grid(True, alpha=0.7)
    axes[0].legend(loc='best', fontsize=8, ncol=3)
    
    axes[1].set_ylabel('$T$ [K]', fontsize=16)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_ylim(bounds['temp'])
    axes[1].grid(True, alpha=0.7)

    axes[2].set_ylabel('$A_V$ [mag]', fontsize=16)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_xlabel('Time [yr]', fontsize=16)
    axes[2].set_ylim(bounds['av'])
    axes[2].grid(True, alpha=0.7)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(savedirmain / ff / 'ev_all_params_time.pdf', 
                   bbox_inches='tight', dpi=300)
    plt.show()

def plot_all_params_radius(all_data, bounds, particle_IDs, save=True):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), dpi=300, sharex=True)
    
    # Generate colors for all particles
    colors = plt.cm.Oranges(np.linspace(0.2, 1, len(particle_IDs)))

    print(bounds['r'])
    
    for i, pid in enumerate(particle_IDs):
        data = all_data[pid]
        t = data['TIME']
        r = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)

        # Density
        axes[0].plot(r, data['DENSITY'], color=colors[i], lw=2, 
                    label=f'{pid}')
        
        # Temperature
        axes[1].plot(r, data['TEMP'], color=colors[i], lw=2, 
                    label=f'{pid}')

        # A_V
        axes[2].plot(r, data['AV'], color=colors[i], lw=2, 
                    label=f'{pid}')
        
        # C2H2 abundance
        axes[3].plot(r, data['C2H2'], color=colors[i], lw=2, 
                    label=f'{pid}')

    # Format axes
    axes[0].set_ylabel('$\\rho$ [cm$^{-3}$]', fontsize=16)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(bounds['r'])
    axes[0].set_ylim(bounds['density'])
    axes[0].grid(True, alpha=0.7)
    axes[0].legend(loc='best', fontsize=8, ncol=3)
    
    axes[1].set_ylabel('$T$ [K]', fontsize=16)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_ylim(bounds['temp'])
    axes[1].grid(True, alpha=0.7)

    axes[2].set_ylabel('$A_V$ [mag]', fontsize=16)
    axes[2].set_xscale('log')
    axes[2].set_yscale('log')
    axes[2].set_ylim(bounds['av'])
    axes[2].grid(True, alpha=0.7)

    axes[3].set_ylabel('C$_2$H$_2$ Abundance', fontsize=16)
    axes[3].set_xscale('log')
    axes[3].set_yscale('log')
    axes[3].set_xlabel('Radius [pc]', fontsize=16)
    axes[3].grid(True, alpha=0.7)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(savedirmain / ff / 'ev_all_params_radius.pdf', 
                   bbox_inches='tight', dpi=300)
    plt.show()

def plot_abundances(all_data, particle_IDs, savedirmain, save=True):
    """Plot abundances for given particle IDs."""
    parents = ["He", "CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN", 
    "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS", "Mg", "Na", "Fe"]
    daughters = ['C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N'] # Crich 

    colors_parents = plt.cm.tab20b(np.linspace(0, 1, len(parents)))
    colors_daughters = plt.cm.Oranges(np.linspace(0.3, 0.95, len(daughters)))
    
    for pid in tqdm(particle_IDs, total=len(particle_IDs)):
        data = all_data[pid]
        t = data['TIME']
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
        
        # Parents
        for imol, mol in enumerate(parents):
            axes[0].plot(t, data[mol], color=colors_parents[imol], lw=2, label=mol)
        axes[0].set_ylabel('Abundance (wrt H$_{nuc}$)', fontsize=14)
        axes[0].set_yscale('log')
        axes[0].set_xscale('log')
        axes[0].set_title(f'Particle {pid} - Parent Molecules', fontsize=16)
        axes[0].grid(True, alpha=0.7)
        axes[0].legend(loc='best', fontsize=8, ncol=3)
        
        # Daughters
        for imol, mol in enumerate(daughters):
            axes[1].plot(t, data[mol], color=colors_daughters[imol], lw=2, label=mol)
        axes[1].set_ylabel('Abundance (wrt H$_{nuc}$)', fontsize=14)
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
        axes[1].set_title(f'Particle {pid} - Daughter Molecules', fontsize=16)
        axes[1].set_xlabel('Time [yr]', fontsize=14)
        axes[1].grid(True, alpha=0.7)
        axes[1].legend(loc='best', fontsize=8, ncol=3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(savedirmain / ff / 'abundances' / f'ev_{pid}_abundances.pdf', 
                       bbox_inches='tight', dpi=300)
        # plt.show()

def create_animation(all_data, bounds, particle_IDs, 
                     save=True, max_frames=200, fps=30) -> None:
    """Create and optionally save animation of particle evolution.
    
    Parameters:
    -----------
    max_frames : int, default=200
        Maximum number of frames to include (reduces compilation time)
    fps : int, default=30
        Frames per second for the animation (higher = faster playback)
    """
    time_sorted = bounds['time']
    n_total_frames = len(time_sorted)
    
    print(f"Animation info:")
    # Subsample frames if there are too many (speeds up compilation)
    if n_total_frames > max_frames:
        frame_indices = np.linspace(0, n_total_frames - 1, max_frames, dtype=int)
        time_sorted_subsampled = time_sorted[frame_indices]
        print(f"Subsampling: {n_total_frames} -> {max_frames} frames for faster compilation")
    else:
        time_sorted_subsampled = time_sorted
        frame_indices = np.arange(n_total_frames)
    
    n_frames = len(time_sorted_subsampled)
    
    print(f"Time range: {time_sorted[0]:.2e} - {time_sorted[-1]:.2e} yr")
    print(f"Number of frames: {n_frames}")
    print(f"Playback speed: {fps} fps")

    # Create figure and 3D axis with lower DPI for faster rendering
    fig_anim = plt.figure(figsize=(10, 8), dpi=300)
    ax_anim = fig_anim.add_subplot(111, projection='3d')
    ax_anim.set_xlim(bounds['x'])
    ax_anim.set_ylim(bounds['y'])
    ax_anim.set_zlim(bounds['z'])
    ax_anim.set_xlabel('X [pc]', labelpad=10)
    ax_anim.set_ylabel('Y [pc]', labelpad=10)
    ax_anim.set_zlabel('Z [pc]', labelpad=10)
    ax_anim.tick_params(axis='x', pad=5)
    ax_anim.tick_params(axis='y', pad=5)
    ax_anim.tick_params(axis='z', pad=5)

    # Create scatter plots for each particle
    colors = plt.cm.Oranges(np.linspace(0.2, 1, len(particle_IDs)))
    scatters = {}    
    for i, pid in enumerate(particle_IDs):
        scatters[pid] = ax_anim.scatter([], [], [], c=[colors[i]], s=30, 
                                        alpha=0.7, label=f'{pid}')

    # Create time text object to display current time and frame
    time_text = ax_anim.text2D(0.05, 0.95, '', transform=ax_anim.transAxes, 
                            fontsize=12, verticalalignment='top')
    
    ax_anim.legend(loc='upper right', fontsize=8, ncol=2)
    
    def init():
        """Initialize animation"""
        for scatter in scatters.values():
            scatter._offsets3d = ([], [], [])
        time_text.set_text('')
        return list(scatters.values()) + [time_text]
    
    def update(frame):
        """Update function for animation"""
        current_time = time_sorted_subsampled[frame]
        
        for pid, scatter in scatters.items():
            data = all_data[pid]
            # Find all points up to current time
            mask = data['TIME'] <= current_time
            if mask.any():
                x = data['X'][mask]
                y = data['Y'][mask]
                z = data['Z'][mask]
                scatter._offsets3d = (x, y, z)
            else:
                scatter._offsets3d = ([], [], [])
        
        time_text.set_text(f'Time: {current_time:.2e} yr\nFrame: {frame+1}/{n_frames}')
        return list(scatters.values()) + [time_text]
    
    print("Creating animation...")
    anim = FuncAnimation(fig_anim, update, frames=n_frames, init_func=init,
                        blit=False, interval=1000//fps, repeat=True)  # interval in ms
    
    if save:
        # Save animation as MP4 video
        output_path = savedirmain / ff / 'ev_all_3D.mp4'
        try:
            writer = FFMpegWriter(fps=fps, bitrate=1800, codec='libx264')
            anim.save(output_path, writer=writer)
            print("Animation saved as MP4")
        except Exception as e:
            print(f"FFmpeg not available ({e}), saving as GIF instead...")
            output_path = savedirmain / ff / 'ev_all_3D.gif'
            writer = PillowWriter(fps=fps)
            anim.save(output_path, writer=writer)
            print("Animation saved as GIF")

    fig_anim.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.tight_layout()
    plt.savefig(savedirmain / ff / 'ev_all_3D.pdf', dpi=300)

def main():
    global savedirmain, mf, of, ff
    savedirmain = Path('/Users/sam/Documents/GitHub/Chemistry')
    mf = 'evolving_model'
    of = 'ev_output'
    ff = 'figures/Evolution_traces'

    particle_IDs = [
        5001,
        13775,
        22549,
        31323,
        40097,
        48871,
        57645,
        66419,
        75193,
        83967,
        92741,
        101515,
        110289,
        119063,
        127837
    ]

    # Load all particle data
    all_data, bounds = load_all_particles(particle_IDs)
    
    # Plot rho, T, and A_V for all particles
    # plot_all_params_time(all_data, bounds, particle_IDs, save=True)

    # Plot rho, T, A_V, and C2H2 abundance vs radius for all particles
    plot_all_params_radius(all_data, bounds, particle_IDs, save=True)

    # Plot abundances for each particle
    # plot_abundances(all_data, particle_IDs, savedirmain, save=True)

    # Create and save animation (faster: max_frames=200, fps=30)
    # create_animation(all_data, bounds, particle_IDs, save=True, max_frames=200, fps=30)

if __name__ == '__main__':
    main()