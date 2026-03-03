"""Shared molecule list definitions for the Chemistry project."""

# Core parent molecules (C-rich, no He)
# Used in: convert_trace__run_models.py, convert_trace__run_models_parallel.py,
#          inspect_particles.py, plot_evolution_all.py
parents = ["CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN",
           "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS"]

# Parent molecules with He, Mg, Na, Fe (extended)
# Used in: run_plot_models_Mdot_Vinf_var.py
parents_He_extended = ["He", "CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN",
                       "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS", "Mg", "Na", "Fe"]

# Daughter molecules — C-rich (includes CN)
# Used in: convert_trace__run_models.py, convert_trace__run_models_parallel.py,
#          inspect_particles.py, plot_evolution_all.py
daughters_Crich = ['CN', 'C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N']

# Daughter molecules — C-rich (excludes CN)
# Used in: run_plot_single_config_Mdot_Vinf.py (active),
#          run_plot_models_Mdot_Vinf_var.py (commented-out alternative)
daughters_Crich_no_CN = ['C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N']

# Daughter molecules — O-rich
# Used in: run_plot_models_Mdot_Vinf_var.py (active),
#          run_plot_single_config_Mdot_Vinf.py (commented-out alternative)
daughters_Orich = ["SiN", "SiC", "OH", "CN", "SiOH+"]

# Secondary and tertiary daughters, grains, atoms — evolving model
# Used in: convert_trace__run_models.py, convert_trace__run_models_parallel.py
daughters_2 = ["HCO+", "CH", "CH2", "CH3", "NH", "NH2", "SO", "SO2", "HS", "HCNH+", "NH4+"]
daughters_3 = ["H2CO", "H2CS", "CH3CN", "SiC", "SiN"]
grains = ["GSiO", "GH2O", "GC2H2", "GHCN", "GH2S", "GCH4", "GC2H4"]
atoms = ["C", "N", "H", "O", "S", "Si", "Cl", "F", "P"]
atoms_plus = ["C+", "N+", "H+", "O+", "S+", "Si+", "Cl+", "P+"]

# Molecules for plot_molecules() in plot_evolution_all.py
molecules_plot = ['CO', 'CH4', 'C2H2', 'HCN', 'C2H4']

# Parent-daughter pairs for plot_parent_daughter() in plot_evolution_all.py
pd_parents = ["C2H2", "HCN"]
pd_daughters = ['C2H', 'CN']
