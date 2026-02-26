import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
# Set plt font to LaTeX
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times New Roman",
    "font.sans-serif": "helvetica"
})
from pathlib import Path
from tqdm import tqdm
import shutil
from astropy import units as u
from numpy.lib import recfunctions as rfn  # noqa: F401  (kept for potential future use)
from beckers.Chemistry.code.deprecated.convert_trace__run_models import select_particle_ids
from n_distinct_colours import generate_colormap
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (BASE_PATH, parents, daughters_Crich, daughters_Orich,
                    molecules_plot, pd_parents, pd_daughters)

daughters = daughters_Crich

def load_all_particles(particle_IDs):
    """Load all particle trace data and compute global bounds.

    X, Y, Z, R are read directly from the ev_output files.  No coordinate
    conversion is needed because postprocess_ev_output already wrote them
    in the requested units (cm when to_cm=True, pc otherwise).
    """
    all_data = {}
    for pid in particle_IDs:
        filepath = savedirmain / mf / of / f"ev_{pid}.dat"
        data = np.genfromtxt(filepath, comments="#", skip_header=4, names=True)
        # Ensure data is always at least 1D (handles single-row files)
        if data.ndim == 0:
            data = np.array([data])
        all_data[pid] = data

    # Collect all data across all particles (already in correct units)
    all_x = np.concatenate([data['X'] for data in all_data.values()])
    all_y = np.concatenate([data['Y'] for data in all_data.values()])
    all_z = np.concatenate([data['Z'] for data in all_data.values()])
    all_r = np.concatenate([data['R'] for data in all_data.values()])
    all_times = np.concatenate([data['TIME'] for data in all_data.values()])
    all_density = np.concatenate([data['DENSITY'] for data in all_data.values()])
    all_temp = np.concatenate([data['TEMP'] for data in all_data.values()])
    all_av = np.concatenate([data['AV'] for data in all_data.values()])

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
    axes[0].set_ylabel('$n$ [cm$^{-3}$]', fontsize=16)
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

def plot_all_params_radius(all_data, bounds, particle_IDs, normalize_ab = False, save=True):
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), dpi=300, sharex=True)
    
    # Generate colors for all particles
    colors = plt.cm.Oranges(np.linspace(0.2, 1, len(particle_IDs)))
    colors = generate_colormap(len(particle_IDs)).colors

    ref_data = all_data[particle_IDs[0]]['C2H2']
    
    for i, pid in enumerate(particle_IDs):
        data = all_data[pid]
        if i == 0:
            ref_abundance = data['C2H2']
        r = data['R']

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
        ab = data['C2H2']
        if normalize_ab:
            norm_ab = ab * (ref_abundance[0] / ab[0])
            axes[3].plot(r, norm_ab, color=colors[i], lw=2, 
                    label=f'{pid}')
        else:
            axes[3].plot(r, ab, color=colors[i], lw=2, 
                    label=f'{pid}')

    # Format axes
    axes[0].set_ylabel('$n$ [cm$^{-3}$]', fontsize=16)
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
    axes[3].set_xlabel('Radius [cm]' if to_cm else 'Radius [pc]', fontsize=16)
    axes[3].grid(True, alpha=0.7)
    
    plt.tight_layout()
    
    if save and normalize_ab:
        name = 'ev_all_params_radius_normab_cm.pdf' if to_cm else 'ev_all_params_radius_normab.pdf'
        plt.savefig(savedirmain / ff / name, bbox_inches='tight', dpi=300)
    elif save:
        name = 'ev_all_params_radius_cm.pdf' if to_cm else 'ev_all_params_radius.pdf'
        plt.savefig(savedirmain / ff / name, bbox_inches='tight', dpi=300)
    plt.show()

def plot_molecules(all_data, bounds, particle_IDs, save=True):
    """Plot normalized abundances for CO, CH4, C2H2, HCN, and C2H4."""
    molecules = molecules_plot
    
    fig, axes = plt.subplots(5, 1, figsize=(10, 14), dpi=300, sharex=True)
    
    # Generate colors for all particles
    colors = plt.cm.Oranges(np.linspace(0.2, 1, len(particle_IDs)))
    
    for i, pid in enumerate(particle_IDs):
        data = all_data[pid]
        r = data['R']
        
        for j, mol in enumerate(molecules):
            if mol not in data.dtype.names:
                continue
            abundance = data[mol]
            # Normalize to initial abundance
            normalized_ab = abundance / abundance[0]
            
            axes[j].plot(r, abundance, color=colors[i], lw=2, 
                        label=f'{pid}')
    
    # Format axes
    for j, mol in enumerate(molecules):
        axes[j].set_ylabel(f'{mol}', fontsize=14)
        axes[j].set_xscale('log')
        axes[j].set_yscale('log')
        axes[j].set_xlim(bounds['r'])
        axes[j].grid(True, alpha=0.7)
        if j == 0:
            axes[j].legend(loc='best', fontsize=8, ncol=3)
    
    axes[-1].set_xlabel('Radius [cm]' if to_cm else 'Radius [pc]', fontsize=16)
    
    plt.tight_layout()
    
    if save:
        name = 'ev_molecules_cm.pdf' if to_cm else 'ev_molecules.pdf'
        plt.savefig(savedirmain / ff / name, bbox_inches='tight', dpi=300)
    plt.show()

def plot_abundances(all_data, particle_IDs, savedirmain, save=True):
    """Plot abundances for given particle IDs."""

    colors_parents = plt.cm.tab20b(np.linspace(0, 1, len(parents)))
    colors_daughters = plt.cm.Oranges(np.linspace(0.3, 0.95, len(daughters)))
    
    for pid in tqdm(particle_IDs, total=len(particle_IDs)):
        data = all_data[pid]
        r = data['R']
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
        
        # Parents
        for imol, mol in enumerate(parents):
            if mol not in data.dtype.names:
                continue
            axes[0].plot(r, data[mol], color=colors_parents[imol], lw=2, label=mol)
        axes[0].set_ylabel('Abundance (wrt H$_{nuc}$)', fontsize=14)
        axes[0].set_yscale('log')
        axes[0].set_xscale('log')
        axes[0].set_title(f'Particle {pid} - Parent Molecules', fontsize=16)
        axes[0].grid(True, alpha=0.7)
        axes[0].legend(loc='best', fontsize=8, ncol=3)
        
        # Daughters
        for imol, mol in enumerate(daughters):
            if mol not in data.dtype.names:
                continue
            axes[1].plot(r, data[mol], color=colors_daughters[imol], lw=2, label=mol)
        axes[1].set_ylabel('Abundance (wrt H$_{nuc}$)', fontsize=14)
        axes[1].set_yscale('log')
        axes[1].set_xscale('log')
        axes[1].set_title(f'Particle {pid} - Daughter Molecules', fontsize=16)
        axes[1].set_xlabel('Radius [cm]' if to_cm else 'Radius [pc]', fontsize=14)
        axes[1].grid(True, alpha=0.7)
        axes[1].legend(loc='best', fontsize=8, ncol=3)
        
        plt.tight_layout()
        
        if save:
            suffix = '_cm' if to_cm else ''
            outpath = savedirmain / ff / 'abundances' / f'ev_{pid}_abundances{suffix}.pdf'
            plt.savefig(outpath, bbox_inches='tight', dpi=300)
        # plt.show()

def plot_parent_daughter(all_data, bounds, particle_IDs, save=True):
    """Plot parent-daughter abundances for C2H2 -> C2H and HCN -> CN."""
    parents = pd_parents
    daughters = pd_daughters

    fig, axes = plt.subplots(1, len(parents), figsize=(14,7), dpi=300, sharey=True)
    
    # Generate colors for all particles
    colors = plt.cm.Oranges(np.linspace(0.2, 1, len(particle_IDs)))
    # colors = plt.cm.tab20b(np.linspace(0, 1, len(particle_IDs)))
    particle_handles = [
        Line2D([0], [0], color=colors[i], lw=2, label=f'{pid}')
        for i, pid in enumerate(particle_IDs)
    ]
    pd_handles = [
        Line2D([0], [0], color='k', lw=2.5, ls='-', label='Parent'),
        Line2D([0], [0], color='k', lw=1.5, ls=':', label='Daughter')
    ]
    
    for i, pid in enumerate(particle_IDs):
        data = all_data[pid]
        r = data['R']
        
        for j, (par, dau) in enumerate(zip(parents, daughters)):
            if par not in data.dtype.names or dau not in data.dtype.names:
                continue
            ab_par = data[par]
            ab_dau = data[dau]
            
            axes[j].plot(r, ab_par, color=colors[i], lw=2,
                        label=f'{pid}')
            axes[j].plot(r, ab_dau, color=colors[i], lw=1.5, ls=':')
    
    # Format axes
    for j,(par, dau) in enumerate(zip(parents, daughters)):
        axes[j].set_xscale('log')
        axes[j].set_yscale('log')
        axes[j].set_xlim(bounds['r'])
        axes[j].grid(True, alpha=0.7)
        particle_legend = axes[j].legend(handles=particle_handles, loc='best', fontsize=10, ncol=3)
        axes[j].add_artist(particle_legend)
        axes[j].legend(handles=pd_handles, loc='upper right', fontsize=10)
        axes[j].set_xlabel('Radius [cm]' if to_cm else 'Radius [pc]', fontsize=16)
        axes[j].set_title(f'{par} - {dau}', fontsize=16)
        if j==0:
            axes[j].set_ylabel('Abundance (wrt H$_{nuc}$)', fontsize=16)
    
    plt.tight_layout()
    
    if save:
        name = 'ev_parent_daughter_cm.pdf' if to_cm else 'ev_parent_daughter.pdf'
        plt.savefig(savedirmain / ff / name, bbox_inches='tight', dpi=300)
    plt.show()



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
    unit_label = 'cm' if to_cm else 'pc'

    # Format tick labels with scientific notation for 3D axes
    if to_cm:
        ax_anim.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1e}'))
        ax_anim.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1e}'))
        ax_anim.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1e}'))

    ax_anim.set_xlabel(f'X [{unit_label}]', labelpad=10)
    ax_anim.set_ylabel(f'Y [{unit_label}]', labelpad=10)
    ax_anim.set_zlabel(f'Z [{unit_label}]', labelpad=10)
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

                if to_cm:
                    x = (x * u.pc).to(u.cm).value
                    y = (y * u.pc).to(u.cm).value
                    z = (z * u.pc).to(u.cm).value

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

def setup_figure_directories(base_path, figure_folder):
    """Create or clear figure directories for evolution traces and abundances.

    Returns
    -------
    bool
        True if setup successful, False if user cancelled
    """
    directories = [figure_folder, f'{figure_folder}/abundances']

    for dir_name in directories:
        dir_path = base_path / dir_name

        if dir_path.exists():
            if any(dir_path.iterdir()):
                print(f"\nDirectory '{dir_name}' already exists and contains files.")
                response = input(f"Empty '{dir_name}' directory? [y/n]: ").strip().lower()

                if response in {'y', 'yes'}:
                    print(f"   Removing all files in {dir_name}/...")
                    shutil.rmtree(dir_path)
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"   ✓ {dir_name}/ cleared")
                else:
                    print(f"✗ Keeping existing files in {dir_name}/")
            else:
                print(f"✓ {dir_name}/ exists (empty)")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created {dir_name}/")

    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Plot evolution traces for Crich/Orich chemistry outputs.')
    parser.add_argument('--chemistry', default='Crich', choices=['Crich', 'Orich'])
    parser.add_argument('--pmf', default='wind_v10')
    parser.add_argument('--max-particle-id', type=int, default=300000)
    parser.add_argument('--start-index', type=int, default=2)
    parser.add_argument('--n-select', type=int, default=15)
    return parser.parse_args()


def main():
    global savedirmain, mf, of, ff, to_cm
    global daughters
    args = parse_args()
    chemistry_type = args.chemistry
    daughters = daughters_Orich if chemistry_type == 'Orich' else daughters_Crich

    to_cm = True
    savedirmain = BASE_PATH
    pmf = args.pmf
    mf = f'evolving_model/{chemistry_type}'
    of = 'ev_output'
    ff = f'figures/Evolution_traces/{chemistry_type}'

    if not setup_figure_directories(savedirmain, ff):
        print("\n✗ Setup cancelled by user")
        return

    particle_IDs_file = savedirmain / 'traces' / pmf / 'particle_IDs.txt'
    tracesf = f'traces/{pmf}/trace_output_with_av'
    trace_dir = savedirmain / tracesf
    particle_IDs = select_particle_ids(
        particle_IDs_file,
        trace_dir,
        max_particle_id=args.max_particle_id,
        start_index=args.start_index,
        n_select=args.n_select,
    )
    print(f"Selected particle IDs: {particle_IDs}")
    particle_IDs[1] = 29823
    particle_IDs[2] = 46371

    # particle_IDs = [0, 1, 2, 3, 4]
    # particle_IDs = [21549, 311143]

    # Load all particle data
    all_data, bounds = load_all_particles(particle_IDs)
    
    # Plot n, T, and A_V for all particles
    plot_all_params_time(all_data, bounds, particle_IDs, save=True)

    # Plot n, T, A_V, and C2H2 abundance vs radius for all particles
    plot_all_params_radius(all_data, bounds, particle_IDs, normalize_ab=False, save=True)
    
    # Plot normalized abundances for selected molecules
    plot_molecules(all_data, bounds, particle_IDs, save=True)
 
    # Plot abundances for each particle
    plot_abundances(all_data, particle_IDs, savedirmain, save=True)

    # Plot parent-daughter abundances
    plot_parent_daughter(all_data, bounds, particle_IDs, save=True)

    # # Create and save animation (faster: max_frames=200, fps=30)
    create_animation(all_data, bounds, particle_IDs, save=True, max_frames=200, fps=30)

if __name__ == '__main__':
    main()