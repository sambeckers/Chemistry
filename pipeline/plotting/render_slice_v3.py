#!/usr/bin/env python3
"""
sph_render.py

Native Sarracen SPH rendering, wrapped in one class (``SPHRenderer``).

Colour limits
-------------
Every plotting method accepts ``vmin``, ``vmax`` and ``decades``.

* ``decades=4``      -> vmax = max of the rendered image, vmin = vmax / 10**4
* ``vmax=...`` only  -> vmin = vmax / 10**decades
* ``vmin=...`` only  -> vmax = vmin * 10**decades
* ``vmin`` and ``vmax`` both given -> used as-is (decades ignored)
* ``decades=0``      -> switch the automatic range off for this call
* ``decades=None``   -> use the renderer's default (``SPHRenderer(decades=...)``)

The limits are taken from the *rendered image* (not the raw particle
values), so they always match what is actually drawn.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import re
from pathlib import Path

import h5py
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import sarracen
from astropy import units as u
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree

BASE_PATH = Path("/fred/oz304/beckers/Chemistry")

# ---- plot_utils (external helper module) ----------------------------------
_spec = importlib.util.spec_from_file_location(
    "plot_utils", BASE_PATH / "code_chem" / "plotting" / "plot_utils.py"
)
_plot_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_plot_utils)
set_plot_style = _plot_utils.set_plot_style
format_species_label = _plot_utils.format_species_label


class SPHRenderer:
    """Render Phantom/Sarracen SPH dumps with chemistry from HDF5 files."""

    # ------------------------------------------------------------------
    # Class-level constants
    # ------------------------------------------------------------------
    MIN_PER_CELL = 1

    PHYS_HDF5_KEY = {
        "density":     "density",
        "temperature": "temp",
        "av":          "av",
    }

    PHYS_OVERVIEW_DEFAULTS = {
        "temperature": dict(cmap="gist_heat", label=r"$\log_{10}(T)$ [K]"),
        "density":     dict(cmap="gist_heat", label=r"$\log_{10}(n)$ [cm$^{-3}$]"),
        "av":          dict(cmap="gist_heat", label=r"$\log_{10}(A_V)$ [mag]"),
    }

    FRACTIONS_3 = [0.10, 0.50, 1.00]
    FRACTIONS_9 = [0.01, 0.05, 0.10, 0.25, 0.33, 0.50, 0.66, 0.80, 1.00]

    MOLECULES = ['CO', 'CH2', 'CH3', 'CH4', 'HCl', 'CH3CN', 'SiO', 'HCN',
                 'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H', 'C4H', 'C6H', 'SiC',
                 'SiN', 'H2CS', 'H2CO', 'N2', 'NH3', 'H2S', 'HCP', 'H2O',
                 'C2H2', 'CS', 'SiC2', 'HF', 'C2H4', 'SiS']
    # PARENTS = ["CO", "N2", "CH4", "H2O", "SiC2", "CS", "C2H2", "HCN", "SiS",
    #            "SiO", "HCl", "C2H4", "NH3", "HCP", "HF", "H2S"]
    PARENTS = ["CH4", "H2O", "SiC2", "CS", "C2H2", "HCN", "SiS",
               "SiO", "HCl", "NH3", "HCP", "H2S"]
    # DAUGHTERS = ['CH2', 'CH3', 'CH3CN', 'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H',
    #              'C4H', 'C6H', 'SiC', 'SiN', 'H2CS', 'H2CO', 'SO', 'SO2']
    DAUGHTERS = ['CH3CN', 'CN', 'HC3N', 'HC5N', 'HC7N', 'C2H',
                'C4H', 'SiC', 'SiN', 'H2CS', 'H2CO', 'SO']
    PARENT_DAUGHTER_PAIRS = [("C2H2", "C2H"), ("HCN", "CN")]

    CONTOUR_COLS = ["molecule", "R_xy_001", "R_xy_050", "R_xz_001", "R_xz_050"]

    _KNUTH_CONST = np.uint64(0x9E3779B97F4A7C15)

    # ------------------------------------------------------------------
    # Construction / dump selection
    # ------------------------------------------------------------------
    def __init__(
        self,
        dump_dir="/fred/oz304/beckers/v10a09_out/output/dumps",
        phantom_dir="/fred/oz304/beckers/v10a09",
        save_dir_base=None,
        fast: bool = False,
        chemistry: str = "Crich",
        dark_mode: bool = False,
        font_size: int = 24,
        decades: float | None = None,
        cache: bool = True,
    ):
        self.dump_dir = Path(dump_dir)
        self.phantom_dir = Path(phantom_dir)
        self.save_dir_base = Path(
            save_dir_base or BASE_PATH / "figures/v10a09_out"
        )
        self.chemistry = chemistry
        self.dark_mode = dark_mode
        self.font_size = font_size
        self.decades = decades
        self.cache = cache
        self.fast = fast

        self.phantom_path: Path | None = None
        self.dump_path: Path | None = None
        self.dump_stem: str | None = None
        self._cache: dict = {}

        set_plot_style(dark_mode)

    def get_all_dumps(self):
        return sorted(self.dump_dir.glob("dump_*.h5"))

    def list_dumps(self):
        for i, p in enumerate(self.get_all_dumps()):
            print(f"[{i:4d}] {p.name}")

    def set_dump(self, index: int | None = None, path: str | Path | None = None):
        """Select the dump all following plot calls will use."""
        if path is not None:
            h5 = Path(path)
        elif index is not None:
            h5 = self.get_all_dumps()[index]
        else:
            raise ValueError("Provide index= or path=")

        stem = h5.stem.split("_", 1)[1]
        self.dump_path = h5
        self.dump_stem = stem
        self.phantom_path = self.phantom_dir / f"wind_{stem}"
        self.clear_cache()
        print(f"Dump set: {h5.name}  ->  {self.phantom_path.name}")
        return self

    def clear_cache(self):
        self._cache = {}

    def _require_dump(self):
        if self.dump_path is None:
            raise RuntimeError("No dump selected: call set_dump(index=...) first.")

    @property
    def time_yr(self) -> float | None:
        self._require_dump()
        return self.read_time_yr(self.dump_path)

    # ------------------------------------------------------------------
    # Deterministic subsampling
    # ------------------------------------------------------------------
    @classmethod
    def particle_keep_mask(cls, ids: np.ndarray, fraction: float,
                           seed: int = 123456789) -> np.ndarray:
        """Boolean mask selecting a deterministic ~fraction of particles
        (same as plot_stats.py so identical IDs are chosen)."""
        if fraction >= 1.0 - 1e-9:
            return np.ones(len(ids), dtype=bool)
        if fraction <= 1e-9:
            return np.zeros(len(ids), dtype=bool)
        h = (ids.astype(np.uint64) ^ np.uint64(seed)) * cls._KNUTH_CONST
        return h < np.uint64(int(fraction * 0xFFFF_FFFF_FFFF_FFFF))

    def set_fast(self, fast: float | None):
        """Turn fast mode on (e.g. 0.1) or off (None)."""
        self.fast = fast
        self.clear_cache()
        return self

    def _eff_fraction(self, fraction: float) -> float:
        """Requested fraction scaled by the fast factor."""
        if self.fast is None or self.fast >= 1.0:
            return fraction
        return fraction * self.fast

    def _subsample(self, sdf, fraction):
        """Apply the (fast-scaled) particle subsample to a dataframe."""
        eff = self._eff_fraction(fraction)
        if eff < 1.0:
            keep = self.particle_keep_mask(sdf["iorig"].to_numpy(dtype=np.int64), eff)
            sdf = sdf[keep]
            print(f"Retained {keep.sum():,} particles (fraction={eff:.4g})")
        return sdf
    # ------------------------------------------------------------------
    # NaN interpolation
    # ------------------------------------------------------------------
    @staticmethod
    def fill_nans_nearest(arr: np.ndarray, max_distance: float = 100.0,
                          smooth_sigma: float = 0) -> np.ndarray:
        """Fill NaN cells near the particle boundary with the nearest value,
        optionally followed by a log-space Gaussian blur."""
        nan_mask = ~np.isfinite(arr)
        if not nan_mask.any() or nan_mask.all():
            return arr

        distances = distance_transform_edt(nan_mask)
        close_mask = nan_mask & (distances <= max_distance)

        if close_mask.any():
            valid_rows, valid_cols = np.where(~nan_mask)
            interp = NearestNDInterpolator(
                np.column_stack([valid_rows, valid_cols]),
                arr[valid_rows, valid_cols],
            )
            fill_rows, fill_cols = np.where(close_mask)
            filled = arr.copy()
            filled[fill_rows, fill_cols] = interp(fill_rows, fill_cols)
        else:
            filled = arr.copy()

        if smooth_sigma > 0:
            has_data = np.isfinite(filled) & (filled > 0)
            log_arr = np.where(has_data, np.log10(filled), 0.0)
            blurred_log = gaussian_filter(log_arr, sigma=smooth_sigma)
            blurred_weight = gaussian_filter(has_data.astype(float),
                                             sigma=smooth_sigma)
            with np.errstate(invalid="ignore", divide="ignore"):
                log_smooth = np.where(blurred_weight > 0,
                                      blurred_log / blurred_weight, np.nan)
            filled = np.where(has_data, 10.0 ** log_smooth, filled)
        return filled

    def interpolate_rendered_image(self, ax):
        """Fill NaN pixels in the first image on ax."""
        if not ax.images:
            return
        img = ax.images[0]
        arr = np.asarray(img.get_array(), dtype=float)
        arr[~np.isfinite(arr)] = np.nan
        arr[arr <= 0] = np.nan
        img.set_array(self.fill_nans_nearest(arr))

    # ------------------------------------------------------------------
    # Names, masks, HDF5 reading
    # ------------------------------------------------------------------
    @staticmethod
    def normalise_name(name: str) -> str:
        """Map a species/molecule name to its HDF5 dataset key."""
        c = name.strip().lower()
        c = c.replace("+", "_plus").replace("-", "_minus").replace("/", "_")
        c = c.replace("(", "_").replace(")", "_").replace(".", "_")
        c = re.sub(r"[^a-z0-9_]+", "_", c)
        return re.sub(r"_+", "_", c).strip("_")

    @classmethod
    def field_mask(cls, arr: np.ndarray, name: str) -> np.ndarray:
        """Validity mask: physical -> finite & >0; species -> finite & 0<x<=1."""
        base = np.isfinite(arr) & (arr > 0)
        if name in cls.PHYS_HDF5_KEY or name == "velocity":
            return base
        return base & (arr <= 1.0)

    @classmethod
    def read_field_from_hdf5(cls, f: h5py.File, name: str) -> np.ndarray:
        p = f["particles"]
        if name in cls.PHYS_HDF5_KEY:
            key = cls.PHYS_HDF5_KEY[name]
            if key in p:
                return p[key][:]
            raise KeyError(
                f"Physical parameter '{name}' -> HDF5 key '{key}' not found.\n"
                f"Available keys: {sorted(p.keys())}"
            )
        key = cls.normalise_name(name)
        if key in p:
            return p[key][:]
        raise KeyError(
            f"Species '{name}' -> normalised key '{key}' not found.\n"
            f"Available keys: {sorted(p.keys())}"
        )

    @staticmethod
    def read_time_yr(dump_path: Path) -> float | None:
        """Simulation time in years from the HDF5 dump (stored in seconds)."""
        try:
            with h5py.File(dump_path, "r") as f:
                for loc in ("time", "header/time", "particles/time"):
                    try:
                        t_s = float(np.asarray(f[loc]).flat[0])
                        return t_s / (365.25 * 24 * 3600)
                    except KeyError:
                        continue
            return None
        except Exception:
            return None

    @staticmethod
    def add_time_label(fig, time_yr: float, fontsize: int = 10,
                       x: float = 0.05, y: float = 0.98) -> None:
        color = matplotlib.rcParams.get("text.color", "black")
        fig.text(x, y, rf"$t = {round(time_yr)}$ yr", ha="left", va="top",
                 fontsize=fontsize, color=color, fontweight="bold",
                 transform=fig.transFigure)

    # ------------------------------------------------------------------
    # Attaching fields to the Sarracen DataFrame
    # ------------------------------------------------------------------
    def assign_chemistry_fields(self, sdf, fields: list[str]):
        """Attach abundance / physical columns from the HDF5 dump to sdf."""
        if not fields:
            return sdf

        sdf_ids = sdf["iorig"].to_numpy(dtype=np.int64)
        raw_data = {}
        with h5py.File(self.dump_path, "r") as f:
            hdf5_ids = f["particles/id"][:].astype(np.int64)
            for name in fields:
                raw_data[name] = self.read_field_from_hdf5(f, name)

        sort_idx = np.argsort(hdf5_ids)
        hdf5_ids_sorted = hdf5_ids[sort_idx]
        ins = np.searchsorted(hdf5_ids_sorted, sdf_ids)
        ins_clamped = np.clip(ins, 0, len(hdf5_ids_sorted) - 1)
        matched = hdf5_ids_sorted[ins_clamped] == sdf_ids

        for name, raw in raw_data.items():
            sorted_vals = raw[sort_idx]
            vals = np.where(matched, sorted_vals[ins_clamped], np.nan)
            sdf[name] = np.where(self.field_mask(vals, name), vals, np.nan)
        return sdf

    def assign_velocity_field(self, sdf):
        """Attach a 'velocity' column (speed, km/s) from vx/vy/vz."""
        unit_velocity = sdf._params["udist"] / sdf._params["utime"]
        speed_cms = np.sqrt(sdf["vx"]**2 + sdf["vy"]**2
                            + sdf["vz"]**2).to_numpy() * unit_velocity
        speed_kms = speed_cms * (1 * u.cm / u.s).to(u.km / u.s).value
        mask = self.field_mask(speed_kms, "velocity")
        sdf["velocity"] = np.where(mask, speed_kms, np.nan)
        print("Assigned 'velocity' column (km/s)")
        return sdf

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    @staticmethod
    def remove_abundance_outliers(sdf, quantity: str, k: int = 16,
                                  log_thresh: float = 2.0):
        """Drop particles far above the median of their k neighbours."""
        coords = sdf[["x", "y", "z"]].to_numpy()
        vals = sdf[quantity].to_numpy(dtype=float)
        _, idxs = cKDTree(coords).query(coords, k=k + 1)
        local_median = np.median(vals[idxs[:, 1:]], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log10(vals / local_median)
        outlier = np.isfinite(log_ratio) & (log_ratio > log_thresh)
        print(f"Outlier filter: removed {outlier.sum():,} particles")
        return sdf[~outlier]

    def prepare_render_dataframe(self, quantity: str, fraction: float = 1.0,
                                 col_dens: bool = False):
        """Load, subsample and clean a Sarracen DataFrame ready for rendering."""
        self._require_dump()
        eff = self._eff_fraction(fraction)
        key = (str(self.phantom_path), quantity, eff, col_dens)
        if self.cache and key in self._cache:
            print(f"(cache) {quantity}, fraction={fraction}")
            return self._cache[key]

        print(f"\nLoading {self.phantom_path.name}")
        sdf, sinks = sarracen.read_phantom(self.phantom_path)

        if quantity == "velocity":
            sdf = self.assign_velocity_field(sdf)
        else:
            sdf = self.assign_chemistry_fields(sdf, [quantity])

        n_total = len(sdf)
        # if fraction < 1.0:
        #     keep = self.particle_keep_mask(
        #         sdf["iorig"].to_numpy(dtype=np.int64), fraction)
        #     sdf = sdf[keep]
        #     print(f"Retained {keep.sum():,} / {n_total:,} particles")

        sdf = self._subsample(sdf, fraction)

        # Optional: multiply abundance by n(H2) [converted to m^-3]
        if col_dens and quantity not in self.PHYS_HDF5_KEY and quantity != "velocity":
            with h5py.File(self.dump_path, "r") as f:
                hdf5_ids = f["particles/id"][:].astype(np.int64)
                dens_raw = f["particles/density"][:]
            sdf_ids = sdf["iorig"].to_numpy(dtype=np.int64)
            sort_idx = np.argsort(hdf5_ids)
            hdf5_ids_sorted = hdf5_ids[sort_idx]
            ins = np.clip(np.searchsorted(hdf5_ids_sorted, sdf_ids),
                          0, len(hdf5_ids_sorted) - 1)
            matched = hdf5_ids_sorted[ins] == sdf_ids
            dens_matched = np.where(matched, dens_raw[sort_idx][ins], np.nan)
            cm3_to_m3 = (1 * u.cm**-3).to(u.m**-3).value
            sdf[quantity] = sdf[quantity] * dens_matched * cm3_to_m3
            print(f"Multiplied {quantity} by n(H2) -> units: m^-3")

        sdf = sdf.dropna(subset=[quantity])
        print(f"Rendering with {len(sdf):,} particles")

        meta = {
            "fraction": fraction,
            "n_particles": len(sdf),
            "dump_name": self.phantom_path.name.split("_", 1)[1],
            "sinks": sinks,
        }
        if self.cache:
            self._cache[key] = (sdf, meta)
        return sdf, meta

    # ------------------------------------------------------------------
    # Labels, ticks, axes
    # ------------------------------------------------------------------
    @staticmethod
    def quantity_label(quantity: str, dens_weight: bool = False,
                       col_dens: bool = False) -> str:
        if quantity == "density":
            return r"$n$ [cm$^{-3}$]"
        if quantity == "temperature":
            return r"$T$ [K]"
        if quantity == "av":
            return r"$A_V$ [mag]"
        if quantity == "velocity":
            return r"$v$ [km s$^{-1}$]"
        if col_dens:
            spec_inner = format_species_label(quantity).strip("$")
            return rf"$n^{{\mathrm{{{spec_inner}}}}}$ [m$^{{-3}}$]"
        return r"Abundance relative to $\mathrm{H}_2$"

    @staticmethod
    def kau_tick_formatter(val, pos):
        """Format AU tick values as 10^3 AU."""
        return f"{val * 1e-3:.1f}"

    def format_render_axes(self, ax, quantity, plane, meta, xlim=None,
                           xsec=None, interpolate=False, q_in_title=True,
                           show_xlabel=True, show_ylabel=True,
                           show_title=True, plot_single=False,
                           small_ticks=None):
        """Standard axis labels, tick style and title for a render axes."""
        if xlim is not None and xlim > 1000:
            xlabel = r"$x$ [$10^3$ au]"
            ylabel = r"$y$ [$10^3$ au]" if plane == "xy" else r"$z$ [$10^3$ au]"
            ax.xaxis.set_major_formatter(FuncFormatter(self.kau_tick_formatter))
            ax.yaxis.set_major_formatter(FuncFormatter(self.kau_tick_formatter))
        else:
            xlabel = r"$x$ [au]"
            ylabel = r"$y$ [au]" if plane == "xy" else r"$z$ [au]"

        fs = self.font_size if not plot_single else self.font_size / 2
        if show_xlabel:
            ax.set_xlabel(xlabel, fontsize=fs)
        if show_ylabel:
            ax.set_ylabel(ylabel, fontsize=fs)

        tick_size = 4 if small_ticks else 8
        ax.tick_params(axis="both", which="major", direction="in",
                       length=tick_size, width=1, colors="white",
                       top=True, right=True, labelsize=fs)
        ax.tick_params(axis="both", which="minor", direction="in",
                       length=tick_size / 2, width=1, colors="white",
                       top=True, right=True)
        label_color = plt.rcParams.get("text.color", "black")
        ax.xaxis.set_tick_params(labelcolor=label_color)
        ax.yaxis.set_tick_params(labelcolor=label_color)
        ax.minorticks_on()
        ax.set_aspect("equal")
        ax.set_facecolor("k")

        if xlim is not None:
            ax.set_xlim(-xlim, xlim)
            ax.set_ylim(-xlim, xlim)

        if show_title:
            interp_tag = " (interpolated)" if interpolate else ""
            if q_in_title:
                title = rf"{format_species_label(quantity)}"
            else:
                title = rf"{int(meta['fraction'] * 100)}\%{interp_tag}"
            if xsec is not None:
                title += rf", cross section at $z$={xsec:.0f} AU"
            if quantity != "velocity":
                ax.set_title(title, fontsize=fs)

    # ------------------------------------------------------------------
    # Colour limits (vmin / vmax / decades)
    # ------------------------------------------------------------------
    def resolve_clim(self, data, vmin=None, vmax=None, decades=None):
        """Return (vmin, vmax) following the rules in the module docstring."""
        decades = self.decades if decades is None else decades
        if not decades:
            return vmin, vmax
        span = 10.0 ** decades
        if vmin is not None and vmax is not None:
            return vmin, vmax
        if vmin is not None:
            return vmin, vmin * span
        if vmax is None:
            good = data[np.isfinite(data) & (data > 0)]
            if good.size == 0:
                return vmin, vmax
            vmax = float(good.max())
        return vmax / span, vmax

    def apply_clim(self, ax, vmin=None, vmax=None, decades=None,
                   log_scale=True):
        """Set the colour limits of the first image on ax; return (lo, hi)."""
        if not ax.images:
            return None
        img = ax.images[0]
        arr = np.ma.filled(np.asanyarray(img.get_array()).astype(float), np.nan)
        if log_scale:
            lo, hi = self.resolve_clim(arr, vmin, vmax, decades)
        else:
            lo, hi = vmin, vmax
        if lo is not None or hi is not None:
            cur = img.get_clim()
            img.set_clim(lo if lo is not None else cur[0],
                         hi if hi is not None else cur[1])
        return img.get_clim()

    # ------------------------------------------------------------------
    # Contours
    # ------------------------------------------------------------------
    @staticmethod
    def extract_contour_radii(contour_set):
        results = {}
        for i, level in enumerate(contour_set.levels):
            segs = contour_set.allsegs[i]
            if not segs:
                results[level] = np.nan
                continue
            pts = np.concatenate(segs, axis=0)
            results[level] = np.mean(np.sqrt(pts[:, 0]**2 + pts[:, 1]**2))
        return results

    @staticmethod
    def draw_radii_arrows(ax, radii, plane="xy"):
        angles = [45, 135, 225, 315]
        for (level, r), angle in zip(radii.items(), angles):
            if np.isnan(r):
                continue
            a = np.radians(angle)
            ax.annotate("", xy=(r * np.cos(a), r * np.sin(a)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="white", lw=1.5))
        return ax

    def save_contour_radii(self, save_path, quantity, plane, radii,
                           contour_values=None):
        """Append/update contour radii in a tab-separated .txt file."""
        levels = sorted(radii.keys())
        if plane == "xy":
            col_map = {levels[0]: "R_xy_001", levels[-1]: "R_xy_050"}
            fill_cols = {"R_xz_001": "--", "R_xz_050": "--"}
        else:
            col_map = {levels[0]: "R_xz_001", levels[-1]: "R_xz_050"}
            fill_cols = {"R_xy_001": "--", "R_xy_050": "--"}

        new_vals = {}
        for level, col in col_map.items():
            r = radii.get(level, np.nan)
            new_vals[col] = f"{r:.1f}" if np.isfinite(r) else "--"

        save_path = Path(save_path)
        rows = []
        if save_path.exists():
            with open(save_path, newline="") as f:
                rows = list(csv.DictReader(f, delimiter="\t"))

        mol_row = next((r for r in rows if r["molecule"] == quantity), None)
        if mol_row is None:
            rows.append({"molecule": quantity, **fill_cols, **new_vals})
        else:
            mol_row.update(new_vals)

        with open(save_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.CONTOUR_COLS, delimiter="\t")
            w.writeheader()
            w.writerows(rows)
        print(f"Contour radii saved -> {save_path}")

    # ------------------------------------------------------------------
    # AGB velocity / wind cone
    # ------------------------------------------------------------------
    @staticmethod
    def compute_velAGB(sdf, sinks):
        """AGB (sink) velocity vector in cm/s."""
        uv = sdf._params["udist"] / sdf._params["utime"]
        return np.array([sinks["vx"][0] * uv, sinks["vy"][0] * uv,
                         sinks["vz"][0] * uv])

    @staticmethod
    def compute_wind_cone_angle(sinks, sdf, v_wind_kms: float = 10.0) -> float:
        """theta = arctan(v_wind / v_orb), measured from the z=0 plane."""
        uv = sdf._params["udist"] / sdf._params["utime"]
        c = (1 * u.cm / u.s).to(u.km / u.s).value
        vx = sinks["vx"][0] * uv * c
        vz = sinks["vz"][0] * uv * c
        v_orb = np.hypot(vx, vz)  # km/s
        return np.arctan2(v_wind_kms, v_orb)

    def draw_wind_cone_lines(self, ax, theta, xlim=None, color="k", lw=1,
                             ls="-"):
        """Draw an X through the origin at +/-theta from the horizontal,
        with an arc + label for the narrow opening angle on the left."""
        if xlim is not None:
            r = xlim
        else:
            xl, yl = ax.get_xlim(), ax.get_ylim()
            r = max(abs(v) for v in (*xl, *yl)) + 100

        dx = -r * np.sin(theta)
        dz = r * np.cos(theta)
        ax.plot([-dx, dx], [-dz, dz], color=color, lw=lw, ls=ls, alpha=0.8)
        ax.plot([-dx, dx], [dz, -dz], color=color, lw=lw, ls=ls, alpha=0.8)

        narrow_deg = 2.0 * (90.0 - np.degrees(theta))
        arc_r = r * 0.18
        half = narrow_deg / 2.0
        ang = np.linspace(np.radians(180.0 - half), np.radians(180.0 + half), 100)
        ax.plot(arc_r * np.cos(ang), arc_r * np.sin(ang),
                color=color, lw=lw * 0.8, alpha=0.9)

        txt = ax.text(-arc_r * 1.4, 0.0, rf"${narrow_deg:.1f}^\circ$",
                      ha="right", va="center", fontsize=self.font_size / 2,
                      color=color)
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])
        return theta, ax

    # ------------------------------------------------------------------
    # Core: render one quantity onto an axes
    # ------------------------------------------------------------------
    def render_quantity(self, sdf, quantity, plane="xy", ax=None, xlim=None,
                        xsec=None, dens_weight=False, cmap="gist_heat",
                        log_scale=True, vmin=None, vmax=None, decades=None,
                        interpolate=False, fraction=1.0, contours=False,
                        hmin=None, radii_save_path=None):
        """Render an SPH quantity onto a Matplotlib axes using Sarracen."""
        if ax is None:
            _, ax = plt.subplots(dpi=300)

        render_kwargs = dict(
            x="x", y="y" if plane == "xy" else "z",
            cmap=cmap, cbar=False, log_scale=log_scale,
            dens_weight=dens_weight, ax=ax, normalize=True, hmin=hmin,
        )
        if xlim is not None:
            render_kwargs["xlim"] = (-xlim, xlim)
            render_kwargs["ylim"] = (-xlim, xlim)
        if xsec is not None:
            render_kwargs["xsec"] = xsec

        sdf.render(quantity, **render_kwargs)

        fast_on = self.fast is not None and self.fast < 1.0
        if interpolate and (fraction < 1.0 or fast_on):
            self.interpolate_rendered_image(ax)

        # vmin/vmax/decades are applied to the finished image
        self.apply_clim(ax, vmin, vmax, decades, log_scale)

        if contours and ax.images:
            img = ax.images[0]
            data = np.asarray(img.get_array())
            dmax, dmin = np.nanmax(data), np.nanmin(data)
            contour_values = [0.01 * dmax, 0.5 * dmax]
            valid = [cv for cv in contour_values if dmin < cv < dmax]
            if valid:
                cs = ax.contour(data, levels=valid, colors="white",
                                linestyles=["--", "-"], linewidths=1.0,
                                origin="image", extent=img.get_extent())
                radii = self.extract_contour_radii(cs)
                for level, r in radii.items():
                    print(f"Level {level:.3e} -> R = {r}")
                self.draw_radii_arrows(ax, radii, plane=plane)
                if radii_save_path is not None:
                    self.save_contour_radii(radii_save_path, quantity, plane,
                                            radii, contour_values)
            else:
                print(f"Skipping contours for {quantity}: levels outside "
                      f"data range [{dmin:.2e}, {dmax:.2e}]")
        return ax

    # ------------------------------------------------------------------
    # Save-path helper
    # ------------------------------------------------------------------
    def build_save_path(self, kind, quantity=None, plane=None, xlim=None,
                        xsec=None, dens_weight=False, col_dens=False,
                        tag=""):
        """Reproduce the original directory / file naming scheme.

        kind: single | compare | phys_overview | velocity_components |
              parent_daughter | grid
        """
        self._require_dump()
        ext = "pdf" if self.dark_mode else "png"
        stem = self.dump_stem
        xsec_tag = f"_xsec{xsec:.0f}" if xsec is not None else ""
        xlim_tag = f"_{xlim:.0f}AU" if xlim is not None else ""
        dw_tag = "_dw" if dens_weight else ""
        cd_tag = "n" if col_dens else ""
        spatial = "zoom" if (xlim is not None and xlim <= 100) else "full"
        base = self.save_dir_base / self.chemistry / "render"
        if self.fast is not None and self.fast < 1.0:
            base = self.save_dir_base / self.chemistry / f"render_fast"

        if kind == "phys_overview":
            d, name = base / "phys_overview", f"phys_{stem}{xlim_tag}"
        elif kind == "velocity_components":
            d = base / "velocity_components"
            name = f"velcomp_{plane}_{stem}{xlim_tag}"
        elif kind == "parent_daughter":
            d = base / "parent_daughter" / spatial
            name = f"render_{plane}_{quantity}_{stem}{xsec_tag}{xlim_tag}"
        elif kind == "grid":
            qs = list(quantity)
            if all(q in self.PARENTS for q in qs):
                mol_tag = "parents"
            elif all(q in self.DAUGHTERS for q in qs):
                mol_tag = "daughters"
            else:
                mol_tag = "molecules"
            d = base / "grid" / mol_tag
            name = (f"grid_{plane}_{cd_tag}{mol_tag}_{stem}"
                    f"{xsec_tag}{dw_tag}{xlim_tag}")
        else:  # single / compare
            mol_type = ("parent" if quantity in self.PARENTS
                        else "daughter" if quantity in self.DAUGHTERS
                        else "other")
            sub = "compare" if kind == "compare" else "single"
            d = base / sub / mol_type / spatial
            name = (f"render_{plane}_{cd_tag}{quantity}_{stem}"
                    f"{xsec_tag}{dw_tag}{xlim_tag}{tag}")

        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}.{ext}"

    def _finish(self, fig, save_path, show):
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # Single render
    # ------------------------------------------------------------------
    def plot_single_render(self, quantity, plane="xy", xlim=None, xsec=None,
                           fraction=1.0, dens_weight=False, interpolate=False,
                           log_scale=True, cmap="gist_heat", vmin=None,
                           vmax=None, decades=None, contours=False,
                           col_dens=False, hmin=None, radii_save_path=None,
                           time_yr=None, save=False, save_path=None,
                           show=True):
        """Render one quantity for the current dump."""
        sdf, meta = self.prepare_render_dataframe(quantity, fraction, col_dens)

        fig, ax = plt.subplots(dpi=300)
        self.render_quantity(
            sdf, quantity, plane=plane, ax=ax, xlim=xlim, xsec=xsec,
            dens_weight=dens_weight, cmap=cmap, log_scale=log_scale,
            vmin=vmin, vmax=vmax, decades=decades, interpolate=interpolate,
            fraction=fraction, contours=contours, hmin=hmin,
            radii_save_path=radii_save_path,
        )
        self.format_render_axes(ax, quantity, plane, meta, xlim=xlim,
                                xsec=xsec, plot_single=True)

        cbar = fig.colorbar(ax.images[0], ax=ax)
        cbar.set_label(self.quantity_label(quantity, dens_weight, col_dens),
                       fontsize=self.font_size / 2)
        cbar.ax.tick_params(labelsize=self.font_size / 2)

        plt.tight_layout()
        if time_yr is not None:
            self.add_time_label(fig, time_yr)

        if save_path is None and save:
            save_path = self.build_save_path(
                "single", quantity, plane, xlim, xsec, dens_weight, col_dens)
        return self._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # Compare fractions
    # ------------------------------------------------------------------
    @staticmethod
    def grid_shape(n: int) -> tuple[int, int]:
        if n <= 4:
            return 1, n
        if n <= 8:
            return 2, 4
        if n <= 9:
            return 3, 3
        return math.ceil(n / 4), 4

    def plot_fraction_compare(self, quantity, plane="xy", xlim=None,
                              xsec=None, dens_weight=False,
                              interpolate=False, log_scale=True,
                              cmap="gist_heat", vmin=None, vmax=None,
                              decades=None, fractions=None, col_dens=False,
                              nine=False, save=False, save_path=None,
                              show=True):
        """Multi-panel render at several particle fractions.

        All panels share the colour limits of the largest-fraction panel.
        """
        if fractions is None:
            fractions = self.FRACTIONS_9 if nine else self.FRACTIONS_3
        fs = self.font_size
        n = len(fractions)
        nrows, ncols = self.grid_shape(n)

        panel_size = 7
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
            dpi=300, sharex=True, sharey=True, squeeze=False)

        all_ax = list(axes.flat)
        for spare in all_ax[n:]:
            spare.set_visible(False)

        cbar_width, cbar_gap, right_margin = 0.015, 0.001, 0.001
        right = 1.0 - cbar_width - cbar_gap - right_margin
        fig.subplots_adjust(right=right, wspace=-0.1, hspace=0.08)

        for idx, (ax, frac) in enumerate(zip(all_ax[:n], fractions)):
            row_idx, col_idx = divmod(idx, ncols)
            sdf, meta = self.prepare_render_dataframe(quantity, frac, col_dens)
            
            self.render_quantity(
                sdf, quantity, plane=plane, ax=ax, xlim=xlim, xsec=xsec,
                dens_weight=dens_weight, cmap=cmap, log_scale=log_scale,
                vmin=vmin, vmax=vmax, decades=0,   # shared limits set below
                interpolate=interpolate, fraction=frac)
            self.format_render_axes(
                ax, quantity, plane, meta, xlim=xlim, xsec=xsec,
                interpolate=interpolate, q_in_title=False,
                show_xlabel=False, show_ylabel=False, show_title=False)
            ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')

            pct = int(frac * 100)
            frac_label = (rf"\textbf{{{pct}\%}}" if pct >= 1
                          else rf"\textbf{{{frac * 100:.1f}\%}}")
            text = ax.text(0.97, 0.97, frac_label, transform=ax.transAxes,
                           ha="right", va="top", fontsize=fs + 6,
                           color="white", fontweight="bold")
            text.set_path_effects([pe.Stroke(linewidth=2.5, foreground='black'),
                                   pe.Normal()])

        # ---- shared colour limits from the reference (largest) fraction ----
        ref_ax = all_ax[fractions.index(max(fractions))]
        clim = self.apply_clim(ref_ax, vmin, vmax, decades, log_scale)
        if clim is not None:
            for ax in all_ax[:n]:
                ax.images[0].set_clim(*clim)

        # ---- one colourbar per row ----------------------------------------
        cbar_left = right + cbar_gap - 0.02
        for row_idx in range(nrows):
            row_axes = [axes[row_idx, c] for c in range(ncols)
                        if axes[row_idx, c].get_visible()]
            if not row_axes:
                continue
            y0 = min(a.get_position().y0 for a in row_axes)
            y1 = max(a.get_position().y1 for a in row_axes)
            cbar_ax = fig.add_axes([cbar_left, y0, cbar_width, y1 - y0])
            cbar = fig.colorbar(row_axes[0].images[0], cax=cbar_ax)
            cbar.set_label(self.quantity_label(quantity, dens_weight, col_dens),
                           fontsize=fs)
            cbar.ax.tick_params(labelsize=fs)

        visible = [a for a in all_ax[:n] if a.get_visible()]
        x_center = 0.5 * (min(a.get_position().x0 for a in visible)
                          + max(a.get_position().x1 for a in visible))
        y_sup = max(a.get_position().y1 for a in visible) + 0.02
        y_sup = min(y_sup, 0.98) if fractions == self.FRACTIONS_9 else 0.95
        fig.suptitle(format_species_label(quantity), fontsize=fs,
                     fontweight="bold", x=x_center, y=y_sup)

        # ---- axis labels: x bottom row, y left column ----------------------
        big = xlim is not None and xlim > 1000
        unit = r'$10^3$ au' if big else 'au'
        xlab = rf'$x$ [{unit}]'
        ylab = rf'$y$ [{unit}]' if plane == "xy" else rf'$z$ [{unit}]'
        for c in range(ncols):
            if axes[nrows - 1, c].get_visible():
                axes[nrows - 1, c].set_xlabel(xlab, fontsize=fs)
        for r in range(nrows):
            if axes[r, 0].get_visible():
                axes[r, 0].set_ylabel(ylab, fontsize=fs)

        if save_path is None and save:
            tag = ("_frac9" if nine else "_frac3") + "_compare"
            save_path = self.build_save_path(
                "compare", quantity, plane, xlim, xsec, dens_weight,
                col_dens, tag=tag)
        return self._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # Physical overview (2 rows x 3 cols)
    # ------------------------------------------------------------------
    def plot_phys_overview(self, xlim=None, log_scale=True, decades=None,
                           save=False, save_path=None, show=True):
        """Temperature / density / Av in xy (top) and xz (bottom).

        Each column shares one colour range (taken from the xy panel) and the
        colourbar above it matches that range.
        """
        self._require_dump()
        fs = self.font_size
        quantities = ["temperature", "density", "av"]
        planes = ["xy", "xz"]

        print(f"\nLoading {self.phantom_path.name} (phys overview)")
        sdf_raw, _ = sarracen.read_phantom(self.phantom_path)
        sdf_raw = self.assign_chemistry_fields(sdf_raw, quantities)
        sdf_raw = sdf_raw.dropna(subset=quantities)
        sdf_raw = self._subsample(sdf_raw, 1.0)
        print(f"Rendering with {len(sdf_raw):,} particles")

        meta = {"fraction": 1.0, "n_particles": len(sdf_raw),
                "dump_name": self.phantom_path.name.split("_", 1)[1]}

        panel_size = 6
        nrows, ncols = 2, 3
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows),
            dpi=300, squeeze=False, sharex=True, sharey=True)

        clims = {}
        for col_idx, quantity in enumerate(quantities):
            cmap = self.PHYS_OVERVIEW_DEFAULTS[quantity]["cmap"]
            for row_idx, plane in enumerate(planes):
                ax = axes[row_idx, col_idx]
                self.render_quantity(
                    sdf_raw, quantity, plane=plane, ax=ax, xlim=xlim,
                    dens_weight=False, cmap=cmap, log_scale=log_scale,
                    decades=0, interpolate=False, fraction=1.0)
                self.format_render_axes(
                    ax, quantity, plane, meta, xlim=xlim, q_in_title=False,
                    show_xlabel=False, show_ylabel=False, show_title=False)
                ax.set_xlabel(''); ax.set_ylabel(''); ax.set_title('')
                if xlim is not None:
                    ax.set_xlim(-xlim, xlim); ax.set_ylim(-xlim, xlim)
                else:
                    ax.set_xlim(sdf_raw['x'].min(), sdf_raw['x'].max())
                    ax.set_ylim(sdf_raw['y'].min(), sdf_raw['y'].max())

            # limits from the xy panel, applied to both rows
            clim = self.apply_clim(axes[0, col_idx], None, None, decades,
                                   log_scale)
            axes[1, col_idx].images[0].set_clim(*clim)
            clims[quantity] = clim

        # ---- axis labels ---------------------------------------------------
        unit = r'$10^3$ au' if (xlim is not None and xlim > 1000) else 'au'
        for c in range(ncols):
            axes[nrows - 1, c].set_xlabel(rf'$x$ [{unit}]', fontsize=fs)
        axes[0, 0].set_ylabel(rf'$y$ [{unit}]', fontsize=fs)
        axes[1, 0].set_ylabel(rf'$z$ [{unit}]', fontsize=fs)

        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.06, top=0.78,
                            wspace=-0.3, hspace=0.08)
        fig.canvas.draw()

        # ---- colourbars above each xy panel --------------------------------
        cbar_height, cbar_gap = 0.022, 0.025
        for col_idx, quantity in enumerate(quantities):
            pos = axes[0, col_idx].get_position()
            cbar_ax = fig.add_axes([pos.x0, pos.y1 + cbar_gap, pos.width,
                                    cbar_height])
            lo, hi = clims[quantity]
            norm = Normalize(vmin=np.log10(lo), vmax=np.log10(hi))
            cbar = fig.colorbar(
                ScalarMappable(norm=norm,
                               cmap=self.PHYS_OVERVIEW_DEFAULTS[quantity]["cmap"]),
                cax=cbar_ax, orientation='horizontal')
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(direction='out', length=5, width=1,
                                colors='k', which='major')
            cbar.ax.tick_params(direction='out', length=2.5, width=1,
                                colors='k', which='minor')
            cbar.ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
            cbar.ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.5))
            cbar.ax.tick_params(axis='x', which='minor', labeltop=False)
            cbar.ax.tick_params(labelsize=fs * 0.75)
            cbar.ax.set_title(self.PHYS_OVERVIEW_DEFAULTS[quantity]["label"],
                              fontsize=fs * 0.85, pad=8)

        t = self.read_time_yr(self.dump_path)
        if t is not None:
            y = axes[0, 0].get_position().y1 + 0.075
            self.add_time_label(fig, t, fontsize=20, x=0.06, y=min(y, 0.98))

        if save_path is None and save:
            save_path = self.build_save_path("phys_overview", xlim=xlim)
        return self._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # Velocity components
    # ------------------------------------------------------------------
    def plot_velocity_components(self, plane="xz", xlim=None, fraction=1.0,
                                 cmap="bwr", vsym=None, wind_angle=False,
                                 v_wind_kms=10.0, time_yr=None, save=False,
                                 save_path=None, show=True):
        """Signed in-plane velocity components side by side (bwr, 0-centred).

        vsym: symmetric colour limit [km/s] for both panels.  Default keeps
        the original hard-coded +/-6 (v_x) and +/-3 (v_z or v_y).
        """
        self._require_dump()
        print(f"\nLoading {self.phantom_path.name} (velocity components)")
        sdf, sinks = sarracen.read_phantom(self.phantom_path)

        uv = sdf._params["udist"] / sdf._params["utime"]
        c = (1 * u.cm / u.s).to(u.km / u.s).value
        comp1, comp2 = "x", ("z" if plane == "xz" else "y")
        sdf["v1_kms"] = sdf[f"v{comp1}"].to_numpy() * uv * c
        sdf["v2_kms"] = sdf[f"v{comp2}"].to_numpy() * uv * c

        if fraction < 1.0:
            # keep = self.particle_keep_mask(
            #     sdf["iorig"].to_numpy(dtype=np.int64), fraction)
            # sdf = sdf[keep]
            sdf = self._subsample(sdf, fraction)
            # print(f"Retained {keep.sum():,} particles")

        fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=300,
                                 sharex=True, sharey=True)
        fig.tight_layout(rect=[0, 0, 0.95, 1])

        theta = (self.compute_wind_cone_angle(sinks, sdf, v_wind_kms)
                 if wind_angle else None)
        lims = {"v1_kms": vsym if vsym else 6, "v2_kms": vsym if vsym else 3}

        for col_name, comp, ax in [("v1_kms", comp1, axes[0]),
                                   ("v2_kms", comp2, axes[1])]:
            render_kwargs = dict(
                x="x", y="y" if plane == "xy" else "z", cmap=cmap,
                cbar=False, log_scale=False, dens_weight=True, ax=ax,
                normalize=True)
            if xlim is not None:
                render_kwargs["xlim"] = (-xlim, xlim)
                render_kwargs["ylim"] = (-xlim, xlim)
            sdf.render(col_name, **render_kwargs)
            ax.images[0].set_clim(-lims[col_name], lims[col_name])

            self.format_render_axes(
                ax, "velocity", plane, {"fraction": fraction}, xlim=xlim,
                show_title=False, plot_single=True, show_xlabel=True,
                show_ylabel=True)

            cbar = fig.colorbar(ax.images[0], ax=ax)
            cbar.set_label(rf"$v_{{{comp}}}$ [km s$^{{-1}}$]",
                           fontsize=self.font_size / 2)
            cbar.ax.tick_params(labelsize=self.font_size / 2)

            ax.tick_params(axis="both", which="major", direction="in",
                           length=4, width=1, colors="k", top=True,
                           right=True, labelsize=self.font_size / 2)
            ax.tick_params(axis="both", which="minor", direction="in",
                           length=2, width=1, colors="k", top=True, right=True)

            if wind_angle:
                self.draw_wind_cone_lines(ax, theta, xlim=xlim, color="k")

        if theta is not None:
            print(f"Wind-cone angle theta = {np.degrees(theta):.2f} deg "
                  f"(v_wind={v_wind_kms} km/s)")
        if time_yr is not None:
            self.add_time_label(fig, time_yr)

        if save_path is None and save:
            save_path = self.build_save_path("velocity_components",
                                             plane=plane, xlim=xlim)
        return self._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # Parent / daughter pair
    # ------------------------------------------------------------------
    def plot_parent_daughter_pair(self, parent, daughter, plane="xy",
                                  xlim=None, xsec=None, dens_weight=False,
                                  interpolate=False, cmap="gist_heat",
                                  vmin=None, vmax=None, decades=None,
                                  col_dens=False, time_yr=None, save=False,
                                  save_path=None, show=True):
        """Parent (left) and daughter (right) on the parent's colour scale."""
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=300,
                                 sharex=True, sharey=True)

        parent_sdf, meta = self.prepare_render_dataframe(
            parent, 1.0, col_dens)
        self.render_quantity(
            parent_sdf, parent, plane=plane, ax=axes[0], xlim=xlim,
            xsec=xsec, dens_weight=dens_weight, cmap=cmap, log_scale=True,
            decades=0, interpolate=interpolate, fraction=1.0)

        daughter_sdf, _ = self.prepare_render_dataframe(
            daughter, 1.0, col_dens)
        self.render_quantity(
            daughter_sdf, daughter, plane=plane, ax=axes[1], xlim=xlim,
            xsec=xsec, dens_weight=dens_weight, cmap=cmap, log_scale=True,
            decades=0, interpolate=interpolate, fraction=1.0)

        # parent sets the scale (with decades if requested); daughter copies
        clim = self.apply_clim(axes[0], vmin, vmax, decades, True)
        parent_img, daughter_img = axes[0].images[0], axes[1].images[0]
        daughter_img.set_clim(*clim)

        for ax, name in zip(axes, (parent, daughter)):
            self.format_render_axes(
                ax, name, plane, meta, xlim=xlim, xsec=xsec,
                q_in_title=False, show_title=False, plot_single=True)
        axes[0].set_title(format_species_label(parent),
                          fontsize=self.font_size / 2)
        axes[1].set_title(format_species_label(daughter),
                          fontsize=self.font_size / 2)
        axes[1].set_ylabel('')

        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.92,
                            wspace=0.1)
        fig.canvas.draw()

        lp, rp = axes[0].get_position(), axes[1].get_position()
        cbar_ax = fig.add_axes([rp.x1 + 0.01, lp.y0, 0.02, lp.y1 - lp.y0])
        cbar = fig.colorbar(parent_img, cax=cbar_ax)
        cbar.set_label(self.quantity_label(parent, dens_weight, col_dens),
                       fontsize=self.font_size / 2)
        cbar.ax.tick_params(labelsize=self.font_size / 2)

        if time_yr is not None:
            self.add_time_label(fig, time_yr)

        if save_path is None and save:
            save_path = self.build_save_path(
                "parent_daughter", f"{parent}_{daughter}", plane, xlim, xsec)
        return self._finish(fig, save_path, show)

    # ------------------------------------------------------------------
    # Molecule grid
    # ------------------------------------------------------------------
    @staticmethod
    def maybe_use_offset_cbar_ticks(cbar, font_size):
        """Compact tick labels (with a x10^N multiplier) when the colourbar
        spans less than one decade."""
        vmin, vmax = cbar.norm.vmin, cbar.norm.vmax
        if vmin is None or vmax is None or vmin <= 0 or vmax <= 0:
            return
        if math.log10(vmax) - math.log10(vmin) >= 1.0:
            return

        exp = math.floor(math.log10(vmax))
        factor = 10.0 ** exp
        subs = [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        ticks = [s * factor for s in subs
                 if vmin * 0.999 <= s * factor <= vmax * 1.001]
        if len(ticks) < 2:
            ticks = list(np.linspace(vmin, vmax, 5)[1:-1])

        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{t / factor:.2g}" for t in ticks])
        cbar.ax.text(0.5, 1.02, rf"$\times10^{{{exp}}}$",
                     transform=cbar.ax.transAxes, ha="center", va="bottom",
                     fontsize=font_size * 0.70, clip_on=False)

    def plot_molecule_grid(self, quantities, plane="xy", xlim=None, xsec=None,
                           dens_weight=False, interpolate=False,
                           log_scale=True, cmap="gist_heat", vmin=None,
                           vmax=None, decades=None, col_dens=False,
                           contours=False, radii_save_path=None,
                           time_yr=None, save=False, save_path=None,
                           show=True):
        """4-column grid of molecules, each panel with its own colourbar
        (and, if decades is set, its own `decades`-wide colour range)."""
        n = len(quantities)
        if n == 0:
            print("plot_molecule_grid: no quantities supplied.")
            return None
        fs = 12

        ncols = 3
        nrows = math.ceil(n / ncols)

        # ---- layout in inches ----------------------------------------------
        panel     = 1.9    # square panel side
        cb_w      = 0.15   # colourbar width
        cb_pad    = 0.05   # panel -> colourbar
        tick_room = 0.50   # colourbar tick labels, before the next panel
        row_gap   = 0.35   # vertical gap (holds the title of the row below)
        left, right, bottom, top = 0.60, 0.35, 0.55, 0.30

        col_pitch = panel + cb_pad + cb_w + tick_room
        fig_w = left + ncols * col_pitch + right
        fig_h = bottom + nrows * panel + (nrows - 1) * row_gap + top
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

        def rect(row, col, x_off, w):
            """Inch-based box -> figure-fraction [x0, y0, w, h]."""
            x0 = left + col * col_pitch + x_off
            y0 = bottom + (nrows - 1 - row) * (panel + row_gap)
            return [x0 / fig_w, y0 / fig_h, w / fig_w, panel / fig_h]

        shared_ax = None
        for i, quantity in enumerate(quantities):
            print(f"Rendering {quantity} ({i + 1}/{n})")
            row, col = divmod(i, ncols)
            kwargs = dict(sharex=shared_ax, sharey=shared_ax) if shared_ax else {}
            ax  = fig.add_axes(rect(row, col, 0, panel), **kwargs)
            cax = fig.add_axes(rect(row, col, panel + cb_pad, cb_w))
            if shared_ax is None:
                shared_ax = ax

            is_bottom = (i + ncols) >= n
            is_left = (col == 0)

            sdf, meta = self.prepare_render_dataframe(quantity, 1.0, col_dens)
            self.render_quantity(
                sdf, quantity, plane=plane, ax=ax, xlim=xlim, xsec=xsec,
                dens_weight=dens_weight, cmap=cmap, log_scale=log_scale,
                vmin=vmin, vmax=vmax, decades=decades,
                interpolate=interpolate, fraction=1.0, contours=contours,
                radii_save_path=radii_save_path)
            self.format_render_axes(
                ax, quantity, plane, meta, xlim=xlim, xsec=xsec,
                show_xlabel=is_bottom, show_ylabel=is_left,
                small_ticks=True, plot_single=True)

            if not is_bottom:
                ax.tick_params(labelbottom=False); ax.set_xlabel("")
            if not is_left:
                ax.tick_params(labelleft=False); ax.set_ylabel("")

            cbar = fig.colorbar(ax.images[0], cax=cax)
            cbar.ax.tick_params(labelsize=fs * 0.85, direction="out",
                                length=2, width=0.8)
            if col == ncols - 1 or i == n - 1:
                cbar.set_label(
                    self.quantity_label(quantity, dens_weight, col_dens),
                    fontsize=fs * 0.9)
            cbar.ax.yaxis.set_minor_locator(mticker.NullLocator())
            self.maybe_use_offset_cbar_ticks(cbar, fs)
            ax.set_title(ax.get_title(), pad=4)

        if time_yr is not None:
            self.add_time_label(fig, time_yr, x=0.03, y=0.995)

        if save_path is None and save:
            save_path = self.build_save_path(
                "grid", quantities, plane, xlim, xsec, dens_weight, col_dens)
        if save_path is not None:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved: {save_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------
    # CSV config reader (for SLURM-style batch runs)
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_bool(val) -> bool:
        return str(val).strip().lower() in ("true", "1", "yes")

    @staticmethod
    def _parse_float_or_none(val):
        v = val.strip()
        return None if v in ("", "None") else float(v)

    @classmethod
    def load_render_config_row(cls, path: str, row_index: int) -> dict:
        """Read one 0-based row of configs_render.csv into a settings dict."""
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not 0 <= row_index < len(rows):
            raise IndexError(f"row {row_index} out of range (0-{len(rows) - 1})")
        row = rows[row_index]

        q_raw = row.get("quantity", "").strip()
        quantity = {"parents": cls.PARENTS, "daughters": cls.DAUGHTERS,
                    "molecules": cls.MOLECULES}.get(
                        q_raw, [] if q_raw == "" else [q_raw])

        def _floats(key):
            raw = row.get(key, "").strip()
            if raw in ("", "None"):
                return [None]
            return [cls._parse_float_or_none(v) for v in raw.split(",")]

        pb = cls._parse_bool
        return dict(
            quantity=quantity,
            plane=[p.strip() for p in row.get("plane", "xy").split(",")
                   if p.strip()],
            xlim=_floats("xlim"),
            xsec=_floats("xsec"),
            dens_weight=pb(row.get("dens_weight", "FALSE")),
            col_dens=pb(row.get("col-dens", "FALSE")),
            mol_grid=pb(row.get("mol-grid", "FALSE")),
            compare_fractions=pb(row.get("compare_fractions", "FALSE")),
            nine_fractions=pb(row.get("nine-fractions", "FALSE")),
            contours=pb(row.get("contours", "FALSE")),
            parent_daughter=pb(row.get("parent-daughter", "FALSE")),
            phys_overview=pb(row.get("phys-overview", "FALSE")),
        )