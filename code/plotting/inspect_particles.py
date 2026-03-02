import numpy as np
import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess
import html as html_lib
import matplotlib as mpl
from astropy import units as u
from numpy.lib import recfunctions as rfn  # noqa: F401  (kept for potential future use)

sys.path.insert(0, str(Path(__file__).parent.parent))

from run_evolving_models import select_particle_ids
from n_distinct_colours import generate_colormap
from plotly.subplots import make_subplots
import plotly.graph_objects as go
kb_path = Path(__file__).parent.parent / 'KoekenBak'
if kb_path.exists():
    sys.path.insert(0, str(kb_path))

try:
    from kb.modeling.tools import CodeIO as KBCodeIO
except Exception:
    KBCodeIO = None

from config import (BASE_PATH, parents, daughters_Crich, daughters_Orich,
                    daughters_2, daughters_3, atoms, atoms_plus)

daughters = daughters_Crich


def get_distinct_colors(n_colors):
    """Get n distinct colors from n_distinct_colours, robust for small n."""
    min_safe = 14
    cmap = generate_colormap(max(n_colors, min_safe))
    return np.array(cmap.colors[:n_colors])


def split_into_panels(items, n_panels=4):
    """Split list into contiguous groups for panel plotting."""
    n_items = len(items)
    if n_panels <= 0:
        return []

    q, r = divmod(n_items, n_panels) 
    sizes = [q + 1 if i < r else q for i in range(n_panels)]
    groups = []
    start = 0
    for size in sizes:
        groups.append(items[start:start + size])
        start += size
    return groups


def resolve_species_field_name(species_name, available_names):
    """Map configured species names to numpy.genfromtxt-safe field names."""
    if species_name in available_names:
        return species_name

    sanitized = species_name.replace('+', '')
    if '+' in species_name:
        plus_candidate = f"{sanitized}_1"
        if plus_candidate in available_names:
            return plus_candidate

    if sanitized in available_names:
        return sanitized

    return None


def find_latest_1d_model_dir(base_path, chemistry_type='Crich'):
    """Find latest complete 1D model directory with chemistry outputs."""
    models_root = base_path / 'output_1D' / f'complete_1D_model_{chemistry_type}' / 'models'
    if not models_root.exists():
        return None

    candidates = [
        path for path in models_root.iterdir()
        if path.is_dir() and path.name.startswith('model_')
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.name)
    for model_dir in reversed(candidates):
        if (model_dir / 'csphyspar_smooth.out').exists() and (model_dir / 'csfrac_smooth.out').exists():
            return model_dir
    return None


def parse_full_analysis_output(file_path):
    """Parse OutputFullAnalysis text file into production/destruction lists."""
    sections = {'production': [], 'destruction': []}
    mode = None
    with open(file_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith('*** main production reactions'):
                mode = 'production'
                continue
            if lower.startswith('*** main destruction reactions'):
                mode = 'destruction'
                continue
            if mode in sections:
                sections[mode].append(line)
    return sections


def find_latest_analysis_plot_dir(base_path, chemistry_type='Crich'):
    """Find latest KoekenBak plot_* directory containing Full_analysis folders."""
    stars_root = base_path / 'output_1D' / f'complete_1D_model_{chemistry_type}' / 'stars' / 'cwleo'
    if not stars_root.exists():
        return None

    candidates = [
        path for path in stars_root.iterdir()
        if path.is_dir() and path.name.startswith('plot_')
    ]
    if not candidates:
        return None

    candidates.sort(key=lambda path: path.name)
    for plot_dir in reversed(candidates):
        if any(child.is_dir() and child.name.startswith('Full_analysis_') for child in plot_dir.iterdir()):
            return plot_dir
    return None


def collect_analysis_plot_index(plot_dir):
    """Collect production/destruction PDF files grouped by model and species."""
    if plot_dir is None:
        return {}

    index = {}
    model_dirs = [
        path for path in plot_dir.iterdir()
        if path.is_dir() and path.name.startswith('Full_analysis_')
    ]
    model_dirs.sort(key=lambda path: path.name)

    for model_dir in model_dirs:
        species_map = {}
        for pdf in sorted(model_dir.glob('*.pdf')):
            stem = pdf.stem
            kind = None
            species = None
            if '_production' in stem:
                species = stem.split('_production')[0]
                kind = 'production'
            elif '_destruction' in stem:
                species = stem.split('_destruction')[0]
                kind = 'destruction'

            if not kind or not species:
                continue
            species_map.setdefault(species, {})[kind] = pdf

        if species_map:
            index[model_dir.name] = species_map

    return index


def load_full_analysis_from_model_dir(model_dir, component='major'):
    """Read all analyse-<component>-*.out files for one model directory."""
    if KBCodeIO is None or model_dir is None or (not model_dir.exists()):
        return {}

    files = sorted(
        model_dir.glob(f'analyse-{component}-*.out'),
        key=lambda path: int(path.stem.split('-')[-1]) if path.stem.split('-')[-1].isdigit() else 10**9,
    )
    full = {}
    for idx, filepath in enumerate(files):
        try:
            full[idx] = KBCodeIO.readAnalysisFile(str(filepath))
        except Exception:
            continue
    return full


def extract_reaction_definitions(sections):
    """Build (reaction_id, full_label) tuples from parsed text sections."""
    result = {'production': [], 'destruction': []}
    for kind in ['production', 'destruction']:
        lines = sections.get(kind, [])
        for line in lines:
            pieces = line.split(maxsplit=1)
            if not pieces:
                continue
            reaction_id = pieces[0]
            label = pieces[1] if len(pieces) > 1 else pieces[0]
            label = label.replace('-->', '→')
            result[kind].append((reaction_id, label))
    return result


def build_reaction_series(full_analysis, species, reaction_id, kind):
    """Get (radius, fraction) points for one reaction across all radii."""
    radii = []
    values = []
    for idx in sorted(full_analysis.keys()):
        entry = full_analysis[idx]
        if species not in entry:
            continue
        radius = entry.get('radius')
        main = entry[species].get('MAIN', [])
        matched_value = None
        for row in main:
            if not row:
                continue
            rid = row[0]
            if str(rid) != str(reaction_id):
                continue
            try:
                value = float(row[-1])
            except Exception:
                continue
            if kind == 'production' and value > 0:
                matched_value = value
                break
            if kind == 'destruction' and value < 0:
                matched_value = value
                break
        if matched_value is not None and radius is not None:
            radii.append(radius)
            values.append(matched_value)
    return radii, values


def build_interactive_analysis_fig(model_name, species, sections, full_analysis):
    """Create interactive side-by-side production/destruction figure."""
    defs = extract_reaction_definitions(sections)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=['Production', 'Destruction'],
        horizontal_spacing=0.03,
    )

    prod_defs = defs.get('production', [])
    dest_defs = defs.get('destruction', [])
    prod_colors = mpl.cm.gist_rainbow(np.linspace(0, 1, max(1, len(prod_defs))))
    dest_colors = mpl.cm.gist_rainbow(np.linspace(0, 1, max(1, len(dest_defs))))
    marker_symbols = [
        'circle', 'square', 'diamond', 'cross', 'x',
        'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right',
        'star', 'hexagon', 'pentagon'
    ]

    for i, (rid, label) in enumerate(prod_defs):
        x, y = build_reaction_series(full_analysis, species, rid, 'production')
        if not x:
            continue
        color = f"rgb({int(prod_colors[i][0]*255)}, {int(prod_colors[i][1]*255)}, {int(prod_colors[i][2]*255)})"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(
                    symbol=marker_symbols[i % len(marker_symbols)],
                    size=10,
                    color=color,
                    line=dict(color='#111111', width=0.9),
                ),
                legendgroup='prod',
                customdata=[label] * len(x),
                hovertemplate='%{customdata}<br>r=%{x:.3e}<br>fraction=%{y:.3e}<extra></extra>',
                meta={'isReactionTrace': True},
            ),
            row=1,
            col=1,
        )

    for i, (rid, label) in enumerate(dest_defs):
        x, y = build_reaction_series(full_analysis, species, rid, 'destruction')
        if not x:
            continue
        color = f"rgb({int(dest_colors[i][0]*255)}, {int(dest_colors[i][1]*255)}, {int(dest_colors[i][2]*255)})"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=2),
                marker=dict(
                    symbol=marker_symbols[i % len(marker_symbols)],
                    size=10,
                    color=color,
                    line=dict(color='#111111', width=0.9),
                ),
                legendgroup='dest',
                customdata=[label] * len(x),
                hovertemplate='%{customdata}<br>r=%{x:.3e}<br>fraction=%{y:.3e}<extra></extra>',
                meta={'isReactionTrace': True},
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(type='log', title_text='Radius [cm]', row=1, col=1)
    fig.update_xaxes(type='log', title_text='Radius [cm]', row=1, col=2)
    fig.update_yaxes(title_text='Fraction of total produced', row=1, col=1)
    fig.update_yaxes(title_text='Fraction of total destroyed', row=1, col=2)
    fig.update_yaxes(exponentformat='power', showexponent='all')

    fig.update_layout(
        template='plotly_white',
        height=760,
        margin=dict(l=54, r=22, t=96, b=42),
        legend=dict(
            orientation='h',
            y=1.12,
            yanchor='bottom',
            x=0,
            xanchor='left',
            font=dict(size=16),
            itemsizing='constant',
            itemwidth=120,
            entrywidthmode='pixels',
            entrywidth=460,
        ),
        title=f'{species} ({model_name})',
    )
    fig.add_annotation(
        x=0.23,
        y=1.20,
        xref='paper',
        yref='paper',
        text='Production reactions',
        showarrow=False,
        font=dict(size=14),
    )
    fig.add_annotation(
        x=0.77,
        y=1.20,
        xref='paper',
        yref='paper',
        text='Destruction reactions',
        showarrow=False,
        font=dict(size=14),
    )
    return fig


def load_1d_interface_payload(base_path, chemistry_type='Crich'):
    """Load 1D abundances and full-analysis outputs for interface page."""
    if KBCodeIO is None:
        return None

    model_dir = find_latest_1d_model_dir(base_path, chemistry_type=chemistry_type)
    if model_dir is None:
        return None

    physpar_file = model_dir / 'csphyspar_smooth.out'
    abund_file = model_dir / 'csfrac_smooth.out'

    try:
        radius = np.array(KBCodeIO.getChemistryPhysPar(str(physpar_file), 'RADIUS'))
        fracs = KBCodeIO.getChemistryAbundances(str(abund_file))
    except Exception:
        return None

    analysis = {}
    for txt_file in sorted(model_dir.glob('OutputFullAnalysis-*-major.txt')):
        species = txt_file.name.replace('OutputFullAnalysis-', '').replace('-major.txt', '')
        try:
            analysis[species] = parse_full_analysis_output(txt_file)
        except Exception:
            continue

    analysis_plot_dir = find_latest_analysis_plot_dir(base_path, chemistry_type=chemistry_type)
    analysis_plot_index = collect_analysis_plot_index(analysis_plot_dir)
    models_root = base_path / 'output_1D' / f'complete_1D_model_{chemistry_type}' / 'models'

    return {
        'model_dir': model_dir,
        'models_root': models_root,
        'radius': radius,
        'fracs': fracs,
        'analysis': analysis,
        'analysis_plot_dir': analysis_plot_dir,
        'analysis_plot_index': analysis_plot_index,
    }


def generate_1d_overview_html(payload, save_path, to_cm=True):
    """Generate dedicated 1D overview page with abundances + analysis outputs."""
    radius = payload['radius']
    fracs = payload['fracs']
    analysis = payload['analysis']
    model_dir = payload['model_dir']
    models_root = payload.get('models_root')
    analysis_plot_index = payload.get('analysis_plot_index', {})

    available_species = set(fracs.dtype.names)
    parent_groups = np.array_split(parents, 3)
    daughters_present = [mol for mol in daughters if mol in available_species]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=['Parents I', 'Parents II', 'Parents III', 'Daughters'],
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    parent_palette = get_distinct_colors(max(1, len(parents)))
    daughter_palette = get_distinct_colors(max(1, len(daughters_present)))

    for panel_idx, parent_group in enumerate(parent_groups):
        row, col = (panel_idx // 2) + 1, (panel_idx % 2) + 1
        for mol in parent_group:
            if mol not in available_species:
                continue
            color_idx = parents.index(mol)
            color = (
                f"rgb({int(parent_palette[color_idx][0]*255)},"
                f"{int(parent_palette[color_idx][1]*255)},"
                f"{int(parent_palette[color_idx][2]*255)})"
            )
            fig.add_trace(
                go.Scatter(
                    x=radius,
                    y=fracs[mol],
                    mode='lines',
                    name=mol,
                    line=dict(color=color, width=1.8),
                    showlegend=True,
                    meta={'isMoleculeTrace': True},
                ),
                row=row,
                col=col,
            )

    for idx, mol in enumerate(daughters_present):
        color = (
            f"rgb({int(daughter_palette[idx][0]*255)},"
            f"{int(daughter_palette[idx][1]*255)},"
            f"{int(daughter_palette[idx][2]*255)})"
        )
        fig.add_trace(
            go.Scatter(
                x=radius,
                y=fracs[mol],
                mode='lines',
                name=mol,
                line=dict(color=color, width=1.8),
                showlegend=True,
                meta={'isMoleculeTrace': True},
            ),
            row=2,
            col=2,
        )

    for row in [1, 2]:
        for col in [1, 2]:
            fig.update_xaxes(type='log', row=row, col=col, title_text='Radius [cm]')
            fig.update_yaxes(type='log', row=row, col=col, title_text='Frac. abundance')

    fig.update_yaxes(exponentformat='power', showexponent='all')

    fig.update_layout(
        template='plotly_white',
        height=940,
        margin=dict(l=56, r=24, t=74, b=54),
        legend=dict(
            orientation='h',
            y=1.07,
            yanchor='bottom',
            x=0,
            xanchor='left',
            font=dict(size=12),
        ),
        title=f"1D abundances ({model_dir.name})",
    )

    plot_div = fig.to_html(
        full_html=False,
        include_plotlyjs='cdn',
        div_id='model-1d-plot',
        default_width='100%',
        default_height='940px',
        config={'responsive': True},
    )

    analysis_plot_html_parts = []

    if analysis_plot_index:
        for model_name in sorted(analysis_plot_index.keys()):
            species_map = analysis_plot_index[model_name]
            model_id = model_name.split('_', 3)[-1] if model_name.count('_') >= 3 else model_name
            model_path = models_root / model_id if models_root is not None else None
            full_analysis = load_full_analysis_from_model_dir(model_path, component='major') if model_path is not None else {}

            if not full_analysis:
                continue

            species_cards = []
            for species in sorted(species_map.keys()):
                sections = analysis.get(species)
                if not sections:
                    continue
                analysis_fig = build_interactive_analysis_fig(model_name, species, sections, full_analysis)
                analysis_div = analysis_fig.to_html(
                    full_html=False,
                    include_plotlyjs=False,
                    div_id=f"analysis-{model_name}-{species}".replace('+', 'plus').replace('/', '_').replace(' ', '_'),
                    default_width='100%',
                    default_height='760px',
                    config={'responsive': True},
                )

                if analysis_div:
                    species_cards.append(
                        f"""
                        <details class=\"analysis-species-card\">
                            <summary>{html_lib.escape(species)}</summary>
                            <div class=\"analysis-plot-panel\">
                                {analysis_div}
                            </div>
                        </details>
                        """
                    )

            if species_cards:
                analysis_plot_html_parts.append(
                    f"""
                    <details class=\"analysis-model-card\" open>
                        <summary>{html_lib.escape(model_name)}</summary>
                        <div class=\"analysis-model-content\">
                            {''.join(species_cards)}
                        </div>
                    </details>
                    """
                )

    analysis_plot_html = ''.join(analysis_plot_html_parts) if analysis_plot_html_parts else '<p>No analysis PDF panels found.</p>'

    cards = []
    for species in sorted(analysis.keys()):
        sections = analysis[species]
        production_items = ''.join(
            f"<li>{html_lib.escape(item)}</li>" for item in sections.get('production', [])
        )
        destruction_items = ''.join(
            f"<li>{html_lib.escape(item)}</li>" for item in sections.get('destruction', [])
        )
        cards.append(
            f"""
            <details class=\"analysis-card\">
                <summary>{html_lib.escape(species)}</summary>
                <div class=\"analysis-columns\">
                    <div>
                        <h4>Production</h4>
                        <ol>{production_items or '<li>None</li>'}</ol>
                    </div>
                    <div>
                        <h4>Destruction</h4>
                        <ol>{destruction_items or '<li>None</li>'}</ol>
                    </div>
                </div>
            </details>
            """
        )

    analysis_html = ''.join(cards) if cards else '<p>No OutputFullAnalysis text outputs found.</p>'

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>1D Model Overview</title>
    <style>
        html {{
            background: #f8f9fa;
            transition: background-color 0.4s ease;
        }}
        html.dark-mode {{
            background: #1e1e1e;
        }}
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            min-height: 100%;
            color: #24292f;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            transition: background-color 0.4s ease, color 0.4s ease;
        }}
        body {{
            background: #f8f9fa;
        }}
        body.dark-mode {{
            background: #1e1e1e;
            color: #d0d7de;
        }}
        body.theme-transitioning .panel-shell,
        body.theme-transitioning .analysis-model-card,
        body.theme-transitioning .analysis-species-card,
        body.theme-transitioning .analysis-plot-panel,
        body.theme-transitioning .analysis-card {{
            transition: background-color 0.4s ease, border-color 0.4s ease !important;
        }}
        .page {{
            min-height: 100vh;
            box-sizing: border-box;
            padding: 12px 14px 14px;
            overflow: auto;
        }}
        .panel-shell {{
            max-width: 1120px;
            margin: 0 auto;
            border: 1px solid rgba(208,215,222,0.8);
            border-radius: 12px;
            background: rgba(255,255,255,0.94);
            padding: 8px 10px;
        }}
        body.dark-mode .panel-shell {{
            border-color: rgba(100,100,100,0.6);
            background: rgba(30,30,30,0.92);
        }}
        .meta {{
            font-size: 13px;
            margin-bottom: 8px;
            color: #57606a;
        }}
        body.dark-mode .meta {{
            color: #9aa4b1;
        }}
        .analysis-wrap {{
            margin-top: 10px;
            border: 1px solid rgba(208,215,222,0.8);
            border-radius: 10px;
            padding: 8px 10px;
            background: rgba(255,255,255,0.9);
        }}
        body.dark-mode .analysis-wrap {{
            border-color: rgba(100,100,100,0.6);
            background: rgba(30,30,30,0.9);
        }}
        .analysis-wrap h3 {{
            margin: 4px 0 10px;
            font-size: 15px;
        }}
        .analysis-model-card,
        .analysis-species-card {{
            border: 1px solid rgba(208,215,222,0.7);
            border-radius: 8px;
            margin-bottom: 8px;
            padding: 6px 8px;
            background: rgba(246,248,250,0.9);
        }}
        body.dark-mode .analysis-model-card,
        body.dark-mode .analysis-species-card {{
            border-color: rgba(100,100,100,0.6);
            background: rgba(45,45,45,0.9);
        }}
        .analysis-model-card > summary,
        .analysis-species-card > summary {{
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
        }}
        .analysis-model-content {{
            margin-top: 8px;
        }}
        .analysis-plot-stack {{
            margin-top: 8px;
            display: grid;
            grid-template-columns: 1fr;
            gap: 10px;
        }}
        .analysis-plot-panel {{
            border: 1px solid rgba(208,215,222,0.75);
            border-radius: 8px;
            padding: 8px;
            background: rgba(255,255,255,0.94);
        }}
        .analysis-plot-panel .js-plotly-plot,
        .analysis-plot-panel .plot-container,
        .analysis-plot-panel .svg-container {{
            background: transparent !important;
        }}
        body.dark-mode .analysis-plot-panel {{
            border-color: rgba(100,100,100,0.65);
            background: rgba(30,30,30,0.94);
        }}
        body.dark-mode .analysis-plot-panel .js-plotly-plot,
        body.dark-mode .analysis-plot-panel .plot-container,
        body.dark-mode .analysis-plot-panel .svg-container {{
            background: rgba(30,30,30,0.94) !important;
        }}
        body.dark-mode .analysis-plot-panel .main-svg rect.bg,
        body.dark-mode .analysis-plot-panel .main-svg > rect:first-child {{
            fill: #1e1e1e !important;
        }}
        .analysis-card {{
            border: 1px solid rgba(208,215,222,0.7);
            border-radius: 8px;
            margin-bottom: 8px;
            padding: 6px 8px;
            background: rgba(246,248,250,0.9);
        }}
        body.dark-mode .analysis-card {{
            border-color: rgba(100,100,100,0.6);
            background: rgba(45,45,45,0.9);
        }}
        .analysis-card summary {{
            cursor: pointer;
            font-weight: 600;
            font-size: 13px;
        }}
        .analysis-columns {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            margin-top: 8px;
        }}
        .analysis-columns h4 {{
            margin: 2px 0 6px;
            font-size: 12px;
        }}
        .analysis-columns ol {{
            margin: 0;
            padding-left: 18px;
            font-size: 11px;
            line-height: 1.35;
            max-height: 220px;
            overflow: auto;
        }}
        @media (max-width: 980px) {{
            .analysis-columns {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class=\"page\">
        <div class=\"meta\">Model source: {html_lib.escape(str(model_dir))}</div>
        <div class="panel-shell">{plot_div}</div>
        <div class="analysis-wrap">
            <h3>Analysis plots</h3>
            {analysis_plot_html}
        </div>
        <div class=\"analysis-wrap\">
            <h3>Main analysis outputs (text)</h3>
            {analysis_html}
        </div>
    </div>
<script>
function updatePlotThemeFor1D(isDark) {{
    const plot = document.getElementById('model-1d-plot');
    if (!plot || !plot.data) return;
    const textColor = isDark ? '#d0d7de' : '#24292f';
    const gridColor = isDark ? 'rgba(255,255,255,0.16)' : 'rgba(0,0,0,0.12)';
    const update = {{
        template: isDark ? 'plotly_dark' : 'plotly_white',
        paper_bgcolor: isDark ? '#1e1e1e' : '#f8f9fa',
        plot_bgcolor: isDark ? '#2a2a2a' : '#ffffff',
        font: {{ color: textColor }},
        legend: {{
            bgcolor: isDark ? 'rgba(30,30,30,0.9)' : 'rgba(255,255,255,0.95)',
            bordercolor: isDark ? 'rgba(100,100,100,0.6)' : 'rgba(208,215,222,1)',
            font: {{ color: textColor, size: 12 }}
        }}
    }};
    const axisUpdates = {{}};
    for (let i = 1; i <= 8; i++) {{
        const xk = i === 1 ? 'xaxis' : `xaxis${{i}}`;
        const yk = i === 1 ? 'yaxis' : `yaxis${{i}}`;
        axisUpdates[`${{xk}}.gridcolor`] = gridColor;
        axisUpdates[`${{xk}}.color`] = textColor;
        axisUpdates[`${{xk}}.title.font.color`] = textColor;
        axisUpdates[`${{yk}}.gridcolor`] = gridColor;
        axisUpdates[`${{yk}}.color`] = textColor;
        axisUpdates[`${{yk}}.title.font.color`] = textColor;
    }}
    Plotly.relayout(plot, Object.assign({{}}, update, axisUpdates));
}}

function attachLegendFocusLogic(plotDiv, selectorMetaKey) {{
    if (!plotDiv || !plotDiv.on || !plotDiv.data) return;
    let selectedTrace = -1;
    const targetIndices = [];
    const originalColors = {{}};

    for (let i = 0; i < plotDiv.data.length; i++) {{
        const trace = plotDiv.data[i];
        const isTargetFamily = !!(trace && trace.meta && trace.meta[selectorMetaKey]);
        if (!isTargetFamily) continue;
        targetIndices.push(i);
        originalColors[i] = (trace && trace.line && trace.line.color) ? trace.line.color : '#1f77b4';
    }}

    function applySelection(targetTrace) {{
        if (!targetIndices.length) return;
        const newColors = [];
        const markerColors = [];
        const newOpacities = [];
        for (let k = 0; k < targetIndices.length; k++) {{
            const traceIndex = targetIndices[k];
            const original = originalColors[traceIndex] || '#1f77b4';
            if (targetTrace === -1 || traceIndex === targetTrace) {{
                newColors.push(original);
                markerColors.push(original);
                newOpacities.push(1.0);
            }} else {{
                newColors.push('#b8b8b8');
                markerColors.push('#b8b8b8');
                newOpacities.push(0.22);
            }}
        }}
        Plotly.restyle(
            plotDiv,
            {{ 'line.color': newColors, 'marker.color': markerColors, 'opacity': newOpacities }},
            targetIndices
        );
    }}

    plotDiv.on('plotly_legendclick', function(event) {{
        const traceIndex = event.curveNumber;
        const trace = plotDiv.data[traceIndex];
        const isTargetFamily = !!(trace && trace.meta && trace.meta[selectorMetaKey]);
        if (!isTargetFamily) return true;
        selectedTrace = (selectedTrace === traceIndex) ? -1 : traceIndex;
        applySelection(selectedTrace);
        return false;
    }});
}}

function updateAnalysisPlotThemes(isDark) {{
    const plots = document.querySelectorAll('.analysis-plot-panel .js-plotly-plot');
    plots.forEach((plot) => {{
        const textColor = isDark ? '#d0d7de' : '#24292f';
        const gridColor = isDark ? 'rgba(255,255,255,0.16)' : 'rgba(0,0,0,0.12)';
        const update = {{
            template: isDark ? 'plotly_dark' : 'plotly_white',
            paper_bgcolor: isDark ? '#1e1e1e' : '#f8f9fa',
            plot_bgcolor: isDark ? '#2a2a2a' : '#ffffff',
            font: {{ color: textColor }},
            legend: {{
                bgcolor: isDark ? 'rgba(30,30,30,0.9)' : 'rgba(255,255,255,0.95)',
                bordercolor: isDark ? 'rgba(100,100,100,0.6)' : 'rgba(208,215,222,1)',
                font: {{ color: textColor, size: 16 }},
                itemsizing: 'constant',
                itemwidth: 120,
                entrywidthmode: 'pixels',
                entrywidth: 460
            }}
        }};
        const axisUpdates = {{}};
        for (let i = 1; i <= 12; i++) {{
            const xk = i === 1 ? 'xaxis' : `xaxis${{i}}`;
            const yk = i === 1 ? 'yaxis' : `yaxis${{i}}`;
            axisUpdates[`${{xk}}.gridcolor`] = gridColor;
            axisUpdates[`${{xk}}.color`] = textColor;
            axisUpdates[`${{xk}}.title.font.color`] = textColor;
            axisUpdates[`${{yk}}.gridcolor`] = gridColor;
            axisUpdates[`${{yk}}.color`] = textColor;
            axisUpdates[`${{yk}}.title.font.color`] = textColor;
        }}
        Plotly.relayout(plot, Object.assign({{}}, update, axisUpdates));
    }});
}}

function initLegendLogicFor1D() {{
    const abundancePlot = document.getElementById('model-1d-plot');
    attachLegendFocusLogic(abundancePlot, 'isMoleculeTrace');
    const analysisPlots = document.querySelectorAll('.analysis-plot-panel .js-plotly-plot');
    analysisPlots.forEach((plot) => attachLegendFocusLogic(plot, 'isReactionTrace'));
}}

window.addEventListener('message', function(event) {{
    if (event.data && event.data.type === 'theme-change') {{
        const isDark = event.data.isDark;
        document.body.classList.add('theme-transitioning');
        if (isDark) {{
            document.body.classList.add('dark-mode');
            document.documentElement.classList.add('dark-mode');
        }} else {{
            document.body.classList.remove('dark-mode');
            document.documentElement.classList.remove('dark-mode');
        }}
        updatePlotThemeFor1D(isDark);
        updateAnalysisPlotThemes(isDark);
        setTimeout(() => document.body.classList.remove('theme-transitioning'), 500);
    }}
}});

window.addEventListener('load', function() {{
    initLegendLogicFor1D();

    // Grid-point badge for the 1D abundance plot
    const abundancePlot1D = document.getElementById('model-1d-plot');
    if (abundancePlot1D) {{
        const badge1D = document.createElement('div');
        badge1D.id = 'grid-badge-1d';
        Object.assign(badge1D.style, {{
            position: 'fixed',
            display: 'none',
            pointerEvents: 'none',
            background: 'rgba(24,24,28,0.88)',
            color: '#e8e8e8',
            fontFamily: 'monospace',
            fontSize: '11px',
            lineHeight: '1.4',
            padding: '3px 9px',
            borderRadius: '5px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
            zIndex: '9999',
            whiteSpace: 'nowrap',
        }});
        document.body.appendChild(badge1D);

        abundancePlot1D.on('plotly_hover', function(evt) {{
            const pts = evt.points;
            if (!pts || !pts.length) return;
            const t = pts[0];
            if (!t.data || !t.data.x) return;
            const idx   = t.pointNumber;
            const total = t.data.x.length;
            const r     = t.data.x[idx];
            const rFmt  = r !== undefined ? r.toExponential(3) : '?';
            badge1D.textContent = `Grid pt ${{idx + 1}} / ${{total}}   r = ${{rFmt}} cm`;
            badge1D.style.display = 'block';
        }});
        abundancePlot1D.on('plotly_unhover', function() {{
            badge1D.style.display = 'none';
        }});
        abundancePlot1D.addEventListener('mousemove', function(e) {{
            if (badge1D.style.display === 'none') return;
            badge1D.style.left = (e.clientX + 14) + 'px';
            badge1D.style.top  = (e.clientY - 30) + 'px';
        }});
    }}
}});

if (window.parent !== window) {{
    window.parent.postMessage({{ type: 'request-theme' }}, '*');
}}
</script>
</body>
</html>
"""

    with open(save_path, 'w') as f:
        f.write(html)

    return True

def load_all_particles(particle_IDs, savedirmain, mf, of, to_cm=True):
    """Load all particle trace data.

    X, Y, Z, R are read directly from the ev_output files.  No coordinate
    conversion is needed here because postprocess_ev_output already wrote them
    in the requested units (cm when to_cm=True, pc otherwise).  The to_cm
    parameter is kept for API compatibility only.
    """
    all_data = {}
    for pid in particle_IDs:
        filepath = savedirmain / mf / of / f"ev_{pid}.dat"
        data = np.genfromtxt(filepath, comments="#", skip_header=4, names=True)
        # Ensure data is always at least 1D (handles single-row files)
        if data.ndim == 0:
            data = np.array([data])
        all_data[pid] = data
    return all_data


def plot_particle_inspection(pid, data, to_cm=True, save_path=None):
    """Create interactive inspection plot for a single particle."""
    panel_positions = [(1, 2), (1, 3), (4, 2), (4, 3)]
    panel_roman = ['I', 'II', 'III', 'IV']
    n_parent_panels = 3
    parents_per_panel = len(parents) // n_parent_panels
    parent_groups = [
        parents[:parents_per_panel],
        parents[parents_per_panel:2*parents_per_panel],
        parents[2*parents_per_panel:]
    ]
    molecule_groups = {
        'parents_daughters': {
            'label': 'Parents + Daughters',
            'panel_titles': ['Parents I', 'Daughters', 'Parents II', 'Parents III'],
            'panel_molecules': [parent_groups[0], daughters, parent_groups[1], parent_groups[2]],
        },
        'daughters_2': {
            'label': 'Daughters 2',
            'panel_titles': [f'Daughters 2 {r}' for r in panel_roman],
            'panel_molecules': split_into_panels(daughters_2, 4),
        },
        'daughters_3': {
            'label': 'Daughters 3',
            'panel_titles': [f'Daughters 3 {r}' for r in panel_roman],
            'panel_molecules': split_into_panels(daughters_3, 4),
        },
        'atoms': {
            'label': 'Atoms + Atoms+',
            'panel_titles': [f'Atoms + Atoms+ {r}' for r in panel_roman],
            'panel_molecules': split_into_panels(atoms + atoms_plus, 4),
        },
    }
    default_group_key = 'parents_daughters'
    
    fig = make_subplots(
        rows=6,
        cols=3,
        subplot_titles=[
            'Density',
            molecule_groups[default_group_key]['panel_titles'][0],
            molecule_groups[default_group_key]['panel_titles'][1],
            'Temperature',
            molecule_groups[default_group_key]['panel_titles'][2],
            molecule_groups[default_group_key]['panel_titles'][3],
            'A_V',
        ],
        specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "xy", "rowspan": 3}, {"type": "xy", "rowspan": 3}],
            [None, None, None],
            [{"type": "xy", "rowspan": 2}, None, None],
            [None, {"type": "xy", "rowspan": 3}, {"type": "xy", "rowspan": 3}],
            [{"type": "xy", "rowspan": 2}, None, None],
            [None, None, None],
        ],
        horizontal_spacing=0.08,
        vertical_spacing=0.06,
        column_widths=[0.12, 0.40, 0.40],
        row_heights=[0.165, 0.165, 0.17, 0.165, 0.165, 0.17],
    )
    
    # Adaptive downsampling for faster page load and Plotly rendering on long traces.
    # Keeps first/last points and evenly samples the remainder.
    max_display_points = 1200
    n_points = len(data)
    if n_points > max_display_points:
        sample_idx = np.unique(np.linspace(0, n_points - 1, max_display_points, dtype=int))
        data_plot = data[sample_idx]
    else:
        data_plot = data

    r = data_plot['R']
    unit_label = 'cm' if to_cm else 'pc'

    def _has_positive_finite(values):
        arr = np.asarray(values)
        return bool(np.any(np.isfinite(arr) & (arr > 0.0)))
    
    # Left column: physical properties (no legend interaction)
    fig.add_trace(
        go.Scattergl(x=r, y=data_plot['DENSITY'], mode='lines', name='Density',
                     line=dict(color='#1f77b4', width=2), showlegend=False,
                     meta={'isPhysical': True}),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=r, y=data_plot['TEMP'], mode='lines', name='Temperature',
                     line=dict(color='#d62728', width=2), showlegend=False,
                     meta={'isPhysical': True}),
        row=3, col=1
    )
    fig.add_trace(
        go.Scattergl(x=r, y=data_plot['AV'], mode='lines', name='A_V',
                     line=dict(color='#2ca02c', width=2), showlegend=False,
                     meta={'isPhysical': True}),
        row=5, col=1
    )

    group_trace_indices = {key: [] for key in molecule_groups}
    group_original_colors = {key: [] for key in molecule_groups}
    group_panel_titles = {key: val['panel_titles'] for key, val in molecule_groups.items()}
    group_tab_labels = {key: val['label'] for key, val in molecule_groups.items()}

    for group_key, group_info in molecule_groups.items():
        panel_molecules = group_info['panel_molecules']
        unique_molecules = [mol for panel in panel_molecules for mol in panel]
        colors = get_distinct_colors(max(1, len(unique_molecules)))
        color_by_molecule = {}
        for idx, mol in enumerate(unique_molecules):
            color_by_molecule[mol] = (
                f"rgb({int(colors[idx][0]*255)}, {int(colors[idx][1]*255)}, {int(colors[idx][2]*255)})"
            )

        for panel_idx, molecules_in_panel in enumerate(panel_molecules):
            row, col = panel_positions[panel_idx]
            for mol in molecules_in_panel:
                field_name = resolve_species_field_name(mol, data_plot.dtype.names)
                if field_name is None:
                    continue
                y_vals = data_plot[field_name]
                if not _has_positive_finite(y_vals):
                    continue
                color = color_by_molecule[mol]
                fig.add_trace(
                    go.Scattergl(
                        x=r,
                        y=y_vals,
                        mode='lines',
                        name=mol,
                        line=dict(color=color, width=2),
                        showlegend=True,
                        visible=(group_key == default_group_key),
                        meta={'isMolecule': True, 'groupKey': group_key},
                    ),
                    row=row,
                    col=col,
                )
                group_trace_indices[group_key].append(len(fig.data) - 1)
                group_original_colors[group_key].append(color)

    # Compute x-axis limits from all molecule traces
    _all_ab_x: list[float] = []
    for trace in fig.data:
        meta = getattr(trace, 'meta', None) or {}
        if isinstance(meta, dict) and meta.get('isMolecule'):
            xd = np.asarray(trace.x, dtype=float)
            valid_x = xd[np.isfinite(xd) & (xd > 0)]
            _all_ab_x.extend(valid_x.tolist())

    if _all_ab_x:
        _ab_xrange = [float(np.log10(min(_all_ab_x))), float(np.log10(max(_all_ab_x)))]
    else:
        _ab_xrange = None

    # Axis formatting
    for row, col in [(1, 1), (3, 1), (5, 1), (1, 2), (4, 2), (1, 3), (4, 3)]:
        fig.update_xaxes(
            type='log', row=row, col=col, showgrid=True, gridcolor='rgba(0,0,0,0.2)',
            automargin=True, title_standoff=2, tickfont=dict(size=9),
            mirror=True, ticks='inside',
        )
        fig.update_yaxes(
            type='log', row=row, col=col, showgrid=True, gridcolor='rgba(0,0,0,0.2)',
            automargin=True, title_standoff=2, tickfont=dict(size=9),
            mirror=True, ticks='inside',
        )

    # Apply x-axis limits to molecule panels (y auto-scales to show all data)
    if _ab_xrange is not None:
        for row, col in [(1, 2), (4, 2), (1, 3), (4, 3)]:
            fig.update_xaxes(range=_ab_xrange, row=row, col=col)

    # Hide redundant x tick labels on top panels to reduce clutter/overlap
    for row, col in [(1, 1), (3, 1), (1, 2), (1, 3)]:
        fig.update_xaxes(showticklabels=False, row=row, col=col)

    fig.update_yaxes(title_text='n [cm^-3]', row=1, col=1)
    fig.update_yaxes(title_text='T [K]', row=3, col=1)
    fig.update_yaxes(title_text='A_V [mag]', row=5, col=1)
    fig.update_xaxes(title_text=f'Radius [{unit_label}]', row=5, col=1)

    fig.update_yaxes(title_text='Frac. ab.', row=1, col=2)
    fig.update_yaxes(title_text='Frac. ab.', row=4, col=2)
    fig.update_yaxes(title_text='Frac. ab.', row=1, col=3)
    fig.update_yaxes(title_text='Frac. ab.', row=4, col=3)
    fig.update_xaxes(title_text=f'Radius [{unit_label}]', row=4, col=2)
    fig.update_xaxes(title_text=f'Radius [{unit_label}]', row=4, col=3)

    fig.update_layout(
        template='plotly_white',
        autosize=True,
        margin=dict(l=56, r=10, t=74, b=120),
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='#ffffff',
        legend=dict(
            orientation='h',
            y=-0.05,
            yanchor='top',
            x=0.0,
            xanchor='left',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(208,215,222,1)',
            borderwidth=1,
            font=dict(size=10, color='#24292f'),
            itemwidth=40,
            itemsizing='constant',
        ),
    )
    fig.update_annotations(font=dict(size=11))
    fig.for_each_annotation(lambda ann: ann.update(yshift=4))

    # Keep axis-title fonts compact to avoid cross-panel overlap
    fig.update_xaxes(title_font=dict(size=11))
    fig.update_yaxes(title_font=dict(size=11))
    fig.update_yaxes(exponentformat='power', showexponent='all')

    if save_path:
        plot_div = fig.to_html(
            full_html=False,
            include_plotlyjs='cdn',
            div_id='particle-plot',
            default_width='100%',
            default_height='100%',
            config={'responsive': True}
        )
        interactivity_script = f"""
<script>
(function() {{
    const plot = document.getElementById('particle-plot');
    const groupTraceIndices = {group_trace_indices};
    const groupOriginalColors = {group_original_colors};
    const groupPanelTitles = {group_panel_titles};
    const groupTabLabels = {group_tab_labels};
    const allMoleculeTraceIndices = Array.from(
        new Set(Object.values(groupTraceIndices).flat())
    );
    const moleculePanelAnnotationIndices = [1, 2, 4, 5];
    let activeGroup = '{default_group_key}';
    const markerStoragePrefix = 'particle-markers::{pid}::';
    const markerStoragePath = (function() {{
        const parts = window.location.pathname.split('/').filter(Boolean);
        if (parts.length >= 2) return parts.slice(-2).join('/');
        if (parts.length === 1) return parts[0];
        return window.location.pathname;
    }})();
    const markerStorageKey = markerStoragePrefix + markerStoragePath;
    let selectedTrace = -1;
    let markerTraceIndices = [];
    let focusedMarkerTrace = -1;

    function getActiveTraceIndices() {{
        return groupTraceIndices[activeGroup] || [];
    }}

    function getActiveOriginalColors() {{
        return groupOriginalColors[activeGroup] || [];
    }}

    function updateMoleculePanelTitles() {{
        const titles = groupPanelTitles[activeGroup] || ['Molecules I', 'Molecules II', 'Molecules III', 'Molecules IV'];
        const relayout = {{}};
        for (let i = 0; i < moleculePanelAnnotationIndices.length; i++) {{
            relayout[`annotations[${{moleculePanelAnnotationIndices[i]}}].text`] = titles[i] || '';
        }}
        Plotly.relayout(plot, relayout);
    }}

    function updateActiveTabButton() {{
        const buttons = document.querySelectorAll('#group-tabs .group-tab');
        buttons.forEach((button) => {{
            if (button.dataset.group === activeGroup) {{
                button.classList.add('active');
            }} else {{
                button.classList.remove('active');
            }}
        }});
    }}

    function buildTabs() {{
        const tabContainer = document.getElementById('group-tabs');
        if (!tabContainer) return;
        const orderedKeys = ['parents_daughters', 'daughters_2', 'daughters_3', 'atoms'];
        const keys = orderedKeys.filter((key) => key in groupTabLabels);
        for (const key of keys) {{
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'group-tab';
            button.dataset.group = key;
            button.textContent = groupTabLabels[key];
            button.addEventListener('click', function() {{
                setActiveGroup(key);
            }});
            tabContainer.appendChild(button);
        }}
        updateActiveTabButton();
    }}

    function setActiveGroup(groupKey) {{
        if (!(groupKey in groupTraceIndices)) return;
        activeGroup = groupKey;
        selectedTrace = -1;
        const visibilityMap = {{}};
        for (let i = 0; i < allMoleculeTraceIndices.length; i++) {{
            visibilityMap[allMoleculeTraceIndices[i]] = false;
        }}
        const activeTraceIndices = getActiveTraceIndices();
        for (let i = 0; i < activeTraceIndices.length; i++) {{
            visibilityMap[activeTraceIndices[i]] = true;
        }}
        for (let i = 0; i < markerTraceIndices.length; i++) {{
            const traceIndex = markerTraceIndices[i];
            const trace = plot.data[traceIndex];
            const groupKeyForMarker = trace && trace.meta && trace.meta.groupKey ? trace.meta.groupKey : activeGroup;
            visibilityMap[traceIndex] = (groupKeyForMarker === activeGroup);
        }}
        const visibilityIndices = Object.keys(visibilityMap).map((k) => Number(k));
        const visibilityValues = visibilityIndices.map((idx) => visibilityMap[idx]);
        Plotly.restyle(plot, {{ 'visible': visibilityValues }}, visibilityIndices).then(function() {{
            updateMoleculePanelTitles();
            updateActiveTabButton();
            applySelection(-1);
            // Update font size + margin.b together in one relayout (no double-tap)
            updateLegendForTab();
        }});
    }}

    function canUseStorage() {{
        try {{
            const testKey = '__marker_storage_test__';
            window.localStorage.setItem(testKey, '1');
            window.localStorage.removeItem(testKey);
            return true;
        }} catch (error) {{
            return false;
        }}
    }}

    function getStoredMarkers() {{
        if (!canUseStorage()) return [];
        try {{
            const legacyPathFlat = window.location.pathname
                .replace('/Crich/', '/')
                .replace('/Orich/', '/');
            const filenameOnly = window.location.pathname.split('/').filter(Boolean).slice(-1)[0] || '';
            const candidateKeys = [
                markerStorageKey,
                markerStoragePrefix + window.location.pathname,
                markerStoragePrefix + legacyPathFlat,
                markerStoragePrefix + filenameOnly,
            ];

            for (let i = 0; i < candidateKeys.length; i++) {{
                const key = candidateKeys[i];
                if (!key) continue;
                const raw = window.localStorage.getItem(key);
                if (!raw) continue;
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed) && parsed.length) {{
                    if (key !== markerStorageKey) {{
                        window.localStorage.setItem(markerStorageKey, raw);
                    }}
                    return parsed;
                }}
            }}

            // Final fallback: scan ALL stored keys with this particle prefix.
            const prefix = markerStoragePrefix;
            for (let i = 0; i < window.localStorage.length; i++) {{
                const key = window.localStorage.key(i);
                if (!key || !key.startsWith(prefix) || key === markerStorageKey) continue;
                const legacyRaw = window.localStorage.getItem(key);
                if (!legacyRaw) continue;
                const legacyParsed = JSON.parse(legacyRaw);
                if (Array.isArray(legacyParsed) && legacyParsed.length) {{
                    window.localStorage.setItem(markerStorageKey, legacyRaw);
                    return legacyParsed;
                }}
            }}
            return [];
        }} catch (error) {{
            return [];
        }}
    }}

    function persistMarkers() {{
        if (!canUseStorage()) return;
        const markerPayload = markerTraceIndices
            .map((traceIndex) => {{
                const trace = plot.data[traceIndex];
                if (!trace || !(trace.meta && trace.meta.isUserMarker)) return null;
                const markerColor = trace.marker && trace.marker.color ? trace.marker.color : '#ff7f0e';
                const label = trace.meta && trace.meta.label ? trace.meta.label : 'Marker';
                const baseTraceIndex = trace.meta && Number.isInteger(trace.meta.baseTraceIndex)
                    ? trace.meta.baseTraceIndex
                    : -1;
                const groupKey = trace.meta && trace.meta.groupKey ? trace.meta.groupKey : activeGroup;
                return {{
                    x: trace.x && trace.x.length ? trace.x[0] : null,
                    y: trace.y && trace.y.length ? trace.y[0] : null,
                    label: label,
                    color: markerColor,
                    baseTraceIndex: baseTraceIndex,
                    groupKey: groupKey,
                    xaxis: trace.xaxis || 'x',
                    yaxis: trace.yaxis || 'y',
                }};
            }})
            .filter((item) => item && item.x !== null && item.y !== null);

        try {{
            window.localStorage.setItem(markerStorageKey, JSON.stringify(markerPayload));
        }} catch (error) {{
            // Ignore storage write errors (quota/security restrictions)
        }}
    }}

    function rebuildMarkerIndexList() {{
        markerTraceIndices = [];
        for (let i = 0; i < plot.data.length; i++) {{
            const trace = plot.data[i];
            if (trace && trace.meta && trace.meta.isUserMarker) {{
                markerTraceIndices.push(i);
            }}
        }}
    }}

    function formatScientific(value) {{
        const num = Number(value);
        if (!Number.isFinite(num)) return String(value);
        return num.toExponential(3);
    }}

    function buildMarkerTrace(marker) {{
        return {{
            x: [marker.x],
            y: [marker.y],
            mode: 'markers+text',
            text: [marker.label],
            textposition: 'top center',
            textfont: {{ size: 10, color: '#111111' }},
            name: marker.label,
            showlegend: false,
            cliponaxis: false,
            marker: {{
                color: marker.color,
                size: 10,
                symbol: 'diamond',
                line: {{ color: '#111111', width: 1 }}
            }},
            hovertemplate: '<b>%{{text}}</b><br>x=%{{x:.3e}}<br>y=%{{y:.3e}}<extra></extra>',
            xaxis: marker.xaxis || 'x',
            yaxis: marker.yaxis || 'y',
            meta: {{
                isUserMarker: true,
                baseTraceIndex: marker.baseTraceIndex,
                label: marker.label,
                groupKey: marker.groupKey || activeGroup,
            }},
        }};
    }}

    function applyMarkerFocus() {{
        if (!markerTraceIndices.length) return;
        if (focusedMarkerTrace !== -1 && markerTraceIndices.indexOf(focusedMarkerTrace) === -1) {{
            focusedMarkerTrace = -1;
        }}

        const darkMode = document.body.classList.contains('dark-mode');
        const baseTextColor = darkMode ? '#f0f6fc' : '#111111';
        const dimTextColor = darkMode ? 'rgba(240,246,252,0.35)' : 'rgba(17,17,17,0.32)';

        const sizes = [];
        const markerOpacities = [];
        const textColors = [];
        const lineWidths = [];

        for (let i = 0; i < markerTraceIndices.length; i++) {{
            const traceIndex = markerTraceIndices[i];
            const inFocus = focusedMarkerTrace === -1 || traceIndex === focusedMarkerTrace;
            sizes.push(inFocus ? 11 : 8);
            markerOpacities.push(inFocus ? 1.0 : 0.8);
            textColors.push(inFocus ? baseTextColor : dimTextColor);
            lineWidths.push(inFocus ? 1.2 : 0.8);
        }}

        Plotly.restyle(
            plot,
            {{
                'marker.size': sizes,
                'marker.opacity': markerOpacities,
                'textfont.color': textColors,
                'marker.line.width': lineWidths,
            }},
            markerTraceIndices
        );
    }}

    function focusMarker(traceIndex) {{
        if (focusedMarkerTrace === traceIndex) {{
            focusedMarkerTrace = -1;
            applyMarkerFocus();
            return;
        }}

        Plotly.moveTraces(plot, [traceIndex], [plot.data.length - 1]).then(function() {{
            rebuildMarkerIndexList();
            focusedMarkerTrace = plot.data.length - 1;
            applyMarkerFocus();
        }});
    }}

    window.__particleApplyMarkerFocus = applyMarkerFocus;

    function addMarkerFromPoint(point) {{
        const sourceTrace = plot.data[point.curveNumber];
        if (!sourceTrace) return;
        const pointNumber = Number.isInteger(point.pointNumber) ? point.pointNumber : null;
        const clickedX = pointNumber !== null && sourceTrace.x && pointNumber < sourceTrace.x.length
            ? sourceTrace.x[pointNumber]
            : point.x;
        const clickedY = pointNumber !== null && sourceTrace.y && pointNumber < sourceTrace.y.length
            ? sourceTrace.y[pointNumber]
            : point.y;

        const sourceName = sourceTrace.name || `Trace ${{point.curveNumber}}`;
        const markerColor = sourceTrace.line && sourceTrace.line.color
            ? sourceTrace.line.color
            : '#ff7f0e';
        const marker = {{
            x: clickedX,
            y: clickedY,
            label: `${{sourceName}} @ r=${{formatScientific(clickedX)}}`,
            color: markerColor,
            baseTraceIndex: point.curveNumber,
            groupKey: activeGroup,
            xaxis: sourceTrace.xaxis || 'x',
            yaxis: sourceTrace.yaxis || 'y',
        }};

        Plotly.addTraces(plot, buildMarkerTrace(marker)).then(function() {{
            rebuildMarkerIndexList();
            persistMarkers();
            focusedMarkerTrace = plot.data.length - 1;
            applyMarkerFocus();
        }});
    }}

    function removeMarkerAt(traceIndex) {{
        if (focusedMarkerTrace === traceIndex) {{
            focusedMarkerTrace = -1;
        }}
        Plotly.deleteTraces(plot, [traceIndex]).then(function() {{
            rebuildMarkerIndexList();
            persistMarkers();
            applyMarkerFocus();
        }});
    }}

    function restoreMarkers() {{
        const saved = getStoredMarkers();
        if (!saved.length) return;
        const valid = saved.filter((item) => item && item.x !== null && item.y !== null);
        if (!valid.length) return;
        const traces = valid.map((item) => buildMarkerTrace(item));
        Plotly.addTraces(plot, traces).then(function() {{
            rebuildMarkerIndexList();
            applyMarkerFocus();
        }});
    }}

    function applySelection(targetTrace) {{
        const activeTraceIndices = getActiveTraceIndices();
        const activeColors = getActiveOriginalColors();
        if (!activeTraceIndices.length) {{
            applyMarkerFocus();
            return;
        }}

        const newColors = [];
        const newOpacities = [];
        for (let i = 0; i < activeTraceIndices.length; i++) {{
            const traceIndex = activeTraceIndices[i];
            const trace = plot.data[traceIndex];
            const origColor = activeColors[i] || (trace && trace.line ? trace.line.color : '#1f77b4');
            if (targetTrace === -1 || traceIndex === targetTrace) {{
                newColors.push(origColor);
                newOpacities.push(1.0);
            }} else {{
                newColors.push('#b8b8b8');
                newOpacities.push(0.2);
            }}
        }}

        Plotly.restyle(plot, {{'line.color': newColors, 'opacity': newOpacities}}, activeTraceIndices).then(function() {{
            applyMarkerFocus();
        }});
    }}

    plot.on('plotly_legendclick', function(event) {{
        const traceIndex = event.curveNumber;
        if (getActiveTraceIndices().indexOf(traceIndex) === -1) {{
            return true;
        }}

        if (selectedTrace === traceIndex) {{
            selectedTrace = -1;
        }} else {{
            selectedTrace = traceIndex;
        }}
        applySelection(selectedTrace);
        return false;
    }});

    function copyPointValueToClipboard(point) {{
        const hasX = point && typeof point.x !== 'undefined' && point.x !== null;
        if (!hasX) return;

        const valueToCopy = point.x;
        const numericValue = Number(valueToCopy);
        const textToCopy = Number.isFinite(numericValue)
            ? numericValue.toExponential(5)
            : String(valueToCopy);

        if (navigator.clipboard && navigator.clipboard.writeText) {{
            navigator.clipboard.writeText(textToCopy).catch(function() {{
                const el = document.createElement('textarea');
                el.value = textToCopy;
                el.setAttribute('readonly', '');
                el.style.position = 'absolute';
                el.style.left = '-9999px';
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
            }});
            return;
        }}

        const el = document.createElement('textarea');
        el.value = textToCopy;
        el.setAttribute('readonly', '');
        el.style.position = 'absolute';
        el.style.left = '-9999px';
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
    }}

    plot.on('plotly_click', function(event) {{
        const isShiftClick = !!(event.event && event.event.shiftKey);
        const isMetaClick = !!(event.event && event.event.metaKey);
        const point = event.points && event.points.length ? event.points[0] : null;
        if (!point) return;

        if (isMetaClick) {{
            copyPointValueToClipboard(point);
            return;
        }}

        const trace = plot.data[point.curveNumber];
        const isMarker = trace && trace.meta && trace.meta.isUserMarker;
        if (isShiftClick && isMarker) {{
            removeMarkerAt(point.curveNumber);
            return;
        }}
        if (isShiftClick) return;
        if (isMarker) {{
            focusMarker(point.curveNumber);
            return;
        }}
        addMarkerFromPoint(point);
    }});

    function calcLegend(availableWidth, nTraces) {{
        // Returns {{ fontSize, estimatedHeight }} that fills space without overflow.
        const symWidth  = 44;    // px for colour swatch / line symbol
        const charWidth = 0.62;  // px per char per font-size pt
        const avgLen    = 7;     // avg molecule name length (chars)
        const rowH      = 1.65;  // em row-height multiplier
        const pad       = 22;    // legend border + padding overhead (px)
        const maxBudget = 360;
        for (let fs = 15; fs >= 8; fs--) {{
            const itemW  = symWidth + avgLen * fs * charWidth;
            const perRow = Math.max(1, Math.floor(availableWidth / itemW));
            const rows   = Math.ceil(nTraces / perRow);
            const legH   = rows * Math.ceil(fs * rowH + 4) + pad;
            if (legH <= maxBudget) return {{ fontSize: fs, estimatedHeight: legH }};
        }}
        // Fall through at min font
        const itemW  = symWidth + avgLen * 8 * charWidth;
        const perRow = Math.max(1, Math.floor(availableWidth / itemW));
        const rows   = Math.ceil(nTraces / perRow);
        const legH   = rows * Math.ceil(8 * rowH + 4) + pad;
        return {{ fontSize: 8, estimatedHeight: Math.min(legH, maxBudget) }};
    }}

    let _lastAvailWidth = 900;
    let _lastTopMargin  = 74;
    let _lastW = 900;
    let _lastH = 600;

    function fitPlotToViewport() {{
        const w = Math.max(900, window.innerWidth - 4);
        const h = Math.max(540, window.innerHeight - 4);
        _lastW = w;
        _lastH = h;
        const tabBar = document.getElementById('molecule-tab-bar');
        _lastTopMargin = Math.max(68, (tabBar ? tabBar.offsetHeight : 0) + 18);
        _lastAvailWidth = w - 56 - 10; // l + r margins

        const nTraces = plot.data.filter(
            (t) => t.visible !== false && t.meta && t.meta.isMolecule && t.meta.groupKey === activeGroup
        ).length || 12;
        const {{ fontSize, estimatedHeight }} = calcLegend(_lastAvailWidth, nTraces);

        Plotly.relayout(plot, {{
            width: w,
            height: h,
            'margin.t': _lastTopMargin,
            'margin.b': estimatedHeight,
            'legend.font.size': fontSize,
        }});
    }}

    function updateLegendForTab() {{
        // Tab switch: update only font.size + margin.b in a single call —
        // no width/height change, so charts don't flash or re-layout fully.
        const nTraces = plot.data.filter(
            (t) => t.visible !== false && t.meta && t.meta.isMolecule && t.meta.groupKey === activeGroup
        ).length || 12;
        const {{ fontSize, estimatedHeight }} = calcLegend(_lastAvailWidth, nTraces);
        Plotly.relayout(plot, {{
            'margin.b': estimatedHeight,
            'legend.font.size': fontSize,
        }});
    }}

    // ── Grid-point badge ──────────────────────────────────────────────────────
    // When the user hovers on any subplot, show which radial shell (grid point)
    // they are on.  Plotly snaps to the nearest data point, so pts[0].pointNumber
    // is already the exact 0-based index into the radius array.
    (function() {{
        const badge = document.createElement('div');
        badge.id = 'grid-badge';
        Object.assign(badge.style, {{
            position: 'fixed',
            display: 'none',
            pointerEvents: 'none',
            background: 'rgba(24,24,28,0.88)',
            color: '#e8e8e8',
            fontFamily: 'monospace',
            fontSize: '11px',
            lineHeight: '1.4',
            padding: '3px 9px',
            borderRadius: '5px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
            zIndex: '9999',
            whiteSpace: 'nowrap',
        }});
        document.body.appendChild(badge);

        function getRadiusArray() {{
            for (const t of plot.data) {{
                if (t.meta && t.meta.isPhysical && t.x && t.x.length) return t.x;
            }}
            return null;
        }}

        plot.on('plotly_hover', function(evt) {{
            const pts = evt.points;
            if (!pts || !pts.length) return;
            const radii = getRadiusArray();
            if (!radii) return;
            const idx   = pts[0].pointNumber;   // 0-based shell index
            const total = radii.length;
            const r     = radii[idx];
            const rFmt  = r !== undefined ? r.toExponential(3) : '?';
            badge.textContent = `Grid pt ${{idx + 1}} / ${{total}}   r = ${{rFmt}} cm`;
            badge.style.display = 'block';
        }});

        plot.on('plotly_unhover', function() {{
            badge.style.display = 'none';
        }});

        // Follow the mouse so the badge doesn't obscure the data point
        plot.addEventListener('mousemove', function(e) {{
            if (badge.style.display === 'none') return;
            badge.style.left = (e.clientX + 14) + 'px';
            badge.style.top  = (e.clientY - 30) + 'px';
        }});
    }})();
    // ─────────────────────────────────────────────────────────────────────────

    buildTabs();
    restoreMarkers();
    setActiveGroup(activeGroup);
    setTimeout(fitPlotToViewport, 0);
    window.addEventListener('resize', fitPlotToViewport);
}})();
</script>
"""
        html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Particle {pid}</title>
    <style>
        html, body {{
            margin: 0;
            width: 100%;
            height: 100%;
            background: #f8f9fa;
            overflow: hidden;
            transition: background-color 0.3s ease;
        }}
        body.dark-mode {{
            background: #1e1e1e;
        }}
        body {{
            display: flex;
        }}
        #particle-plot {{
            width: 100vw;
            height: 100vh;
        }}
        .js-plotly-plot,
        .plot-container,
        .svg-container {{
            width: 100% !important;
            height: 100% !important;
        }}
        /* Smooth legend font and opacity transitions on tab switch */
        .js-plotly-plot .legend text {{
            transition: font-size 0.25s ease, opacity 0.25s ease;
        }}
        .js-plotly-plot .traces {{
            transition: opacity 0.2s ease;
        }}
        #molecule-tab-bar {{
            position: fixed;
            top: 2px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 55;
            width: calc(100vw - 18px);
            display: flex;
            justify-content: center;
            pointer-events: none;
        }}
        #group-tabs {{
            pointer-events: auto;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 5px;
            padding: 4px 7px;
            border-radius: 10px;
            background: rgba(255, 255, 255, 0.93);
            border: 1px solid rgba(208, 215, 222, 0.95);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
            max-width: min(1200px, calc(100vw - 24px));
        }}
        .group-tab {{
            border: 1px solid rgba(208, 215, 222, 0.95);
            background: #ffffff;
            color: #24292f;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .group-tab:hover {{
            background: #f0f3f6;
        }}
        .group-tab.active {{
            background: #2f81f7;
            color: #ffffff;
            border-color: #2f81f7;
        }}
        body.dark-mode #group-tabs {{
            background: rgba(30, 30, 30, 0.9);
            border-color: rgba(100, 100, 100, 0.7);
        }}
        body.dark-mode .group-tab {{
            background: #2f343b;
            color: #d0d7de;
            border-color: rgba(100, 100, 100, 0.75);
        }}
        body.dark-mode .group-tab:hover {{
            background: #3a4048;
        }}
        body.dark-mode .group-tab.active {{
            background: #1f6feb;
            border-color: #1f6feb;
            color: #ffffff;
        }}
    </style>
</head>
<body>
{plot_div}
<div id="molecule-tab-bar"><div id="group-tabs"></div></div>
{interactivity_script}
<script>
// Listen for theme changes from parent window
window.addEventListener('message', function(event) {{
    if (event.data && event.data.type === 'theme-change') {{
        const isDark = event.data.isDark;
        if (isDark) {{
            document.body.classList.add('dark-mode');
            updatePlotTheme('plotly_dark', '#1e1e1e', '#2d2d2d');
        }} else {{
            document.body.classList.remove('dark-mode');
            updatePlotTheme('plotly_white', '#f8f9fa', '#ffffff');
        }}
    }}
}});

function updatePlotTheme(template, paperBg, plotBg) {{
    const plot = document.getElementById('particle-plot');
    if (plot && plot.data) {{
        const isDark = template === 'plotly_dark';
        const textColor = isDark ? '#ffffff' : '#24292f';
        const gridColor = isDark ? 'rgba(255,255,255,0.15)' : 'rgba(0,0,0,0.1)';
        
        const update = {{
            'paper_bgcolor': paperBg,
            'plot_bgcolor': plotBg,
            'template': template,
            'legend.bgcolor': isDark ? 'rgba(30,30,30,0.9)' : 'rgba(255,255,255,0.95)',
            'legend.bordercolor': isDark ? 'rgba(100,100,100,0.5)' : 'rgba(208,215,222,1)',
            'legend.font.color': isDark ? '#ffffff' : '#24292f',
            'font.color': textColor
        }};
        

    if (window.__particleApplyMarkerFocus) {{
        window.__particleApplyMarkerFocus();
    }}
        // Update all axes with text colors
        const axisUpdates = {{}};
        for (let i = 1; i <= 18; i++) {{
            const xaxisKey = i === 1 ? 'xaxis' : 'xaxis' + i;
            const yaxisKey = i === 1 ? 'yaxis' : 'yaxis' + i;
            axisUpdates[xaxisKey + '.color'] = textColor;
            axisUpdates[xaxisKey + '.gridcolor'] = gridColor;
            axisUpdates[xaxisKey + '.title.font.color'] = textColor;
            axisUpdates[yaxisKey + '.color'] = textColor;
            axisUpdates[yaxisKey + '.gridcolor'] = gridColor;
            axisUpdates[yaxisKey + '.title.font.color'] = textColor;
        }}
        
        Plotly.relayout(plot, Object.assign({{}}, update, axisUpdates));
    }}
}}

// Check initial theme from parent
if (window.parent !== window) {{
    window.parent.postMessage({{ type: 'request-theme' }}, '*');
}}
</script>
</body>
</html>
"""
        with open(save_path, 'w') as f:
            f.write(html)
    else:
        fig.show()


def generate_html_interface(particle_IDs, output_dir, title="Particle Inspection", has_1d_model=False, chemistry_type='Crich'):
    """Generate an HTML interface to navigate between particle plots."""
    other_chemistry = 'Orich' if chemistry_type == 'Crich' else 'Crich'
    chemistry_targets = {
        chemistry_type: 'index.html',
        other_chemistry: f'../{other_chemistry}/index.html',
    }
    # Both chemistry types are always generated together, so both are always available.
    chemistry_available = {
        chemistry_type: True,
        other_chemistry: True,
    }

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        html, body {{
            height: 100%;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #1e1e1e;
            display: flex;
            flex-direction: column;
            transition: background-color 0.3s ease;
        }}
        :root {{
            --bg: #1e1e1e;
            --surface: #2d2d2d;
            --border: #404040;
            --text: #e0e0e0;
            --muted: #a0a0a0;
            --accent: #4a9eff;
        }}
        body.light-mode {{
            background-color: #f5f5f5;
        }}
        body.light-mode {{
            --bg: #f6f8fa;
            --surface: #ffffff;
            --border: #d0d7de;
            --text: #24292f;
            --muted: #57606a;
            --accent: #0969da;
        }}
        .header {{
            background-color: var(--surface);
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            flex: 0 0 auto;
            z-index: 100;
            display: flex;
            align-items: center;
        }}
        .theme-toggle {{
            background: none;
            border: 1px solid var(--border);
            color: var(--text);
            cursor: pointer;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 18px;
            transition: transform 0.2s ease, background-color 0.2s ease, box-shadow 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
        }}
        .theme-toggle:hover {{
            background-color: var(--border);
            transform: scale(1.05);
        }}
        .theme-toggle:active {{
            transform: scale(0.96);
        }}
        .theme-toggle.ripple-active {{
            animation: themeTogglePop 0.38s cubic-bezier(0.22,0.68,0,1.4);
        }}
        @keyframes themeTogglePop {{
            0%   {{ transform: scale(1); }}
            40%  {{ transform: scale(0.85) rotate(-12deg); }}
            70%  {{ transform: scale(1.18) rotate(8deg); }}
            100% {{ transform: scale(1) rotate(0deg); }}
        }}
        .header-controls {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-left: auto;
            flex: 0 0 auto;
        }}
        .mode-switch {{
            display: inline-flex;
            border: 1px solid var(--border);
            border-radius: 999px;
            overflow: hidden;
            background: var(--bg);
            transition: box-shadow 0.25s ease;
            position: relative;
        }}
        .mode-btn {{
            border: none;
            background: transparent;
            color: var(--text);
            padding: 6px 12px;
            margin: 0;
            border-radius: 0;
            font-size: 12px;
            font-weight: 600;
            box-shadow: none;
            transform: none;
            transition: background-color 0.22s ease, color 0.22s ease, transform 0.18s ease;
            position: relative;
            overflow: hidden;
        }}
        .mode-btn::after {{
            content: '';
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.35) 50%, transparent 60%);
            transform: translateX(-100%);
            transition: none;
            pointer-events: none;
        }}
        .mode-btn.shimmer::after {{
            animation: btnShimmer 0.42s ease forwards;
        }}
        @keyframes btnShimmer {{
            0%   {{ transform: translateX(-120%); }}
            100% {{ transform: translateX(160%); }}
        }}
        .mode-btn:hover {{
            background: var(--border);
            box-shadow: none;
            transform: none;
        }}
        .mode-btn:active {{
            transform: scale(0.97);
        }}
        .mode-btn.active {{
            background: var(--accent);
            color: #ffffff;
        }}
        .mode-btn:disabled {{
            opacity: 0.45;
            cursor: not-allowed;
        }}
        .navigation {{
            background-color: var(--surface);
            padding: 12px 16px;
            border-top: 1px solid var(--border);
            flex: 0 0 auto;
            z-index: 100;
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .marker-help-text {{
            font-size: 11px;
            color: var(--muted);
            white-space: nowrap;
            position: absolute;
            left: 16px;
        }}
        .nav-buttons {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
        }}
        button {{
            background: linear-gradient(135deg, var(--surface) 0%, var(--border) 100%);
            color: var(--text);
            border: 1px solid var(--border);
            padding: 10px 20px;
            margin: 0 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            border-radius: 10px;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }}
        button:hover {{
            background: linear-gradient(135deg, var(--border) 0%, var(--surface) 100%);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
        button:disabled {{
            color: var(--muted);
            background: var(--surface);
            cursor: not-allowed;
            opacity: 0.5;
            transform: none;
            box-shadow: none;
        }}
        .content {{
            max-width: none;
            width: 100%;
            box-sizing: border-box;
            margin: 0 auto;
            padding: 6px 8px 8px;
            background-color: var(--bg);
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
        }}
        .image-container {{
            text-align: center;
            margin: 0;
            flex: 1 1 auto;
            min-height: 0;
            position: relative;
        }}
        .image-container iframe {{
            width: 100%;
            height: 100%;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            min-height: 0;
            transition: opacity 0.34s ease, transform 0.34s ease, filter 0.34s ease;
        }}
        .image-container iframe.loading {{
            opacity: 0;
            transform: scale(0.995);
            filter: blur(1px);
        }}
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--surface);
            display: none;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            gap: 16px;
            border-radius: 6px;
            z-index: 10;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        .loading-overlay.active {{
            display: flex;
            opacity: 1;
            backdrop-filter: blur(2px);
        }}
        .spinner {{
            width: 48px;
            height: 48px;
            border: 4px solid var(--border);
            border-top-color: #ffb38a;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        .loading-text {{
            color: var(--text);
            font-size: 14px;
            font-weight: 500;
            animation: loadingPulse 1.1s ease-in-out infinite;
        }}
        @keyframes loadingPulse {{
            0%, 100% {{ opacity: 0.65; }}
            50% {{ opacity: 1; }}
        }}
        @keyframes buttonTap {{
            0% {{ transform: scale(1); }}
            45% {{ transform: scale(0.94); }}
            100% {{ transform: scale(1); }}
        }}
        .control-tap {{
            animation: buttonTap 0.22s ease;
        }}
        @keyframes rippleOut {{
            0%   {{ opacity: 0.45; transform: scale(0); }}
            100% {{ opacity: 0;    transform: scale(2.8); }}
        }}
        .btn-ripple {{
            position: absolute;
            border-radius: 50%;
            background: rgba(255,255,255,0.55);
            pointer-events: none;
            animation: rippleOut 0.5s ease-out forwards;
        }}
        body.theme-transitioning .header,
        body.theme-transitioning .content,
        body.theme-transitioning .navigation,
        body.theme-transitioning button:not(.particle-chip) {{
            transition: background-color 0.4s ease, color 0.4s ease,
                        border-color 0.4s ease, box-shadow 0.4s ease !important;
        }}
        body.theme-transitioning .particle-chip {{
            transition: background-color 0.4s ease, color 0.4s ease,
                        border-color 0.4s ease, opacity 0.4s ease !important;
        }}
        .info {{
            text-align: center;
            color: var(--muted);
            margin-top: 3px;
            font-size: 13px;
            flex: 0 0 auto;
        }}
        .info p {{
            margin: 0;
        }}
        .legend-wrap {{
            margin-top: 8px;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
            flex: 1 1 auto;
            min-width: 0;
            transition: opacity 0.2s ease;
        }}
        .legend-title {{
            color: var(--text);
            font-size: 13px;
            margin-right: 8px;
            font-weight: 500;
        }}
        .particle-chip {{
            border: 1px solid rgba(255,160,122,0.3);
            border-radius: 16px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 600;
            line-height: 1;
            cursor: pointer;
            background-color: #ffb38a;
            color: #ffffff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            opacity: 0.6;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        body.light-mode .particle-chip {{
            color: #1a1a1a;
            text-shadow: none;
        }}
        .particle-chip:hover {{
            transform: translateY(-2px);
            opacity: 0.85;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }}
        .particle-chip.active {{
            opacity: 1;
            color: #ffffff;
            box-shadow: 0 0 0 3px rgba(255,160,122,0.6), 0 4px 12px rgba(255,160,122,0.4);
            transform: scale(1.05);
        }}
        body.light-mode .particle-chip.active {{
            color: #1a1a1a;
        }}
    </style>
</head>
<body>
    <div class="header">
        <div class="legend-wrap" id="particleLegend">
            <span class="legend-title">Particles:</span>
        </div>
        <div class="header-controls">
            <div class="mode-switch" id="chemistrySwitch">
                <button id="chemCrichBtn" class="mode-btn" onclick="setChemistry('Crich')">Crich</button>
                <button id="chemOrichBtn" class="mode-btn" onclick="setChemistry('Orich')">Orich</button>
            </div>
            <div class="mode-switch" id="modeSwitch">
                <button id="mode3dBtn" class="mode-btn active" onclick="setMode('3D')">3D</button>
                <button id="mode1dBtn" class="mode-btn" onclick="setMode('1D')">1D</button>
            </div>
            <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
                <span id="themeIcon">🌙</span>
            </button>
        </div>
    </div>
    
    <div class="content">
        <div class="image-container">
            <div class="loading-overlay" id="loadingOverlay">
                <div class="spinner"></div>
                <div class="loading-text">Loading particle...</div>
            </div>
            <iframe id="particleFrame" src="" title="Particle inspection plot"></iframe>
        </div>
        <div class="info">
            <p id="particleCount">Particle 1 of 1</p>
        </div>
    </div>
    
    <div class="navigation">
        <div class="marker-help-text">Click a line to add marker &middot; Click marker to focus/unfocus &middot; Shift+click marker to remove &middot; ⌘+click to copy exact value</div>
        <div class="nav-buttons">
            <button id="prevBtn" onclick="changeParticle(-1)">← Previous</button>
            <button id="nextBtn" onclick="changeParticle(1)">Next →</button>
        </div>
    </div>

    <script>
        const particles = """ + str(particle_IDs) + """;
        const has1DModel = """ + str(bool(has_1d_model)).lower() + """;
        const currentChemistry = '""" + chemistry_type + """';
        const chemistryTargets = """ + json.dumps(chemistry_targets) + """;
        const chemistryAvailable = """ + json.dumps(chemistry_available) + """;
        // Pastel orange palette
        const orangePalette = ['#ffb38a', '#ffc5a3', '#ffd7bc', '#ffe0c9', '#ffead6', '#ffcba4', '#ffb891', '#ffa877', '#ff9e6d', '#ffbe9f', '#ffd4b8', '#ffa060', '#ffaa70', '#ffb580', '#ffc090'];
        const legendColors = orangePalette.slice(0, particles.length);
        let currentIndex = 0;
        let currentMode = '3D';

        function updateModeButtons() {
            const mode3d = document.getElementById('mode3dBtn');
            const mode1d = document.getElementById('mode1dBtn');
            if (mode3d) mode3d.classList.toggle('active', currentMode === '3D');
            if (mode1d) {
                mode1d.classList.toggle('active', currentMode === '1D');
                mode1d.disabled = !has1DModel;
            }
        }

        function updateChemistryButtons() {
            const crichBtn = document.getElementById('chemCrichBtn');
            const orichBtn = document.getElementById('chemOrichBtn');
            const buttons = [crichBtn, orichBtn];

            buttons.forEach((btn) => {
                if (!btn) return;
                const chem = btn.id === 'chemCrichBtn' ? 'Crich' : 'Orich';
                btn.classList.toggle('active', chem === currentChemistry);
                btn.disabled = false;
            });
        }

        function setChemistry(chemistry) {
            if (!chemistryAvailable[chemistry]) return;
            if (chemistry === currentChemistry) return;
            const target = chemistryTargets[chemistry];
            if (!target) return;
            window.location.href = target;
        }

        function setMode(mode) {
            if (mode === '1D' && !has1DModel) return;
            if (mode !== '1D' && mode !== '3D') return;
            const clicked = mode === '1D' ? document.getElementById('mode1dBtn') : document.getElementById('mode3dBtn');
            shimmerBtn(clicked);
            currentMode = mode;
            updateModeButtons();
            updateDisplay();
        }

        function shimmerBtn(el) {
            if (!el) return;
            el.classList.remove('shimmer');
            void el.offsetWidth;
            el.classList.add('shimmer');
            el.addEventListener('animationend', () => el.classList.remove('shimmer'), { once: true });
        }

        function popThemeBtn(el) {
            if (!el) return;
            el.classList.remove('ripple-active');
            void el.offsetWidth;
            el.classList.add('ripple-active');
            el.addEventListener('animationend', () => el.classList.remove('ripple-active'), { once: true });
        }

        function animateControlTap(el) {
            if (!el) return;
            el.classList.remove('control-tap');
            void el.offsetWidth;
            el.classList.add('control-tap');
        }
        
        function toggleTheme() {
            const themeBtn = document.querySelector('.theme-toggle');
            popThemeBtn(themeBtn);
            document.body.classList.add('theme-transitioning');
            document.body.classList.toggle('light-mode');
            setTimeout(() => document.body.classList.remove('theme-transitioning'), 500);
            const icon = document.getElementById('themeIcon');
            const isDark = !document.body.classList.contains('light-mode');
            
            if (document.body.classList.contains('light-mode')) {
                icon.textContent = '☀️';
            } else {
                icon.textContent = '🌙';
            }
            
            // Notify iframe about theme change
            const iframe = document.getElementById('particleFrame');
            if (iframe && iframe.contentWindow) {
                iframe.contentWindow.postMessage({ type: 'theme-change', isDark: isDark }, '*');
            }
        }
        
        // Listen for theme requests from iframes
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'request-theme') {
                const isDark = !document.body.classList.contains('light-mode');
                event.source.postMessage({ type: 'theme-change', isDark: isDark }, '*');
            }
        });

        function buildLegend() {
            const legend = document.getElementById('particleLegend');
            particles.forEach((pid, idx) => {
                const chip = document.createElement('button');
                chip.className = 'particle-chip';
                chip.id = `chip-${pid}`;
                chip.textContent = pid;
                chip.style.backgroundColor = legendColors[idx];
                chip.onclick = () => {
                    currentIndex = idx;
                    updateDisplay();
                };
                legend.appendChild(chip);
            });
        }

        function updateDisplay() {
            const pid = particles[currentIndex];
            const iframe = document.getElementById('particleFrame');
            const loadingOverlay = document.getElementById('loadingOverlay');
            const loadingText = document.querySelector('.loading-text');
            const nav = document.querySelector('.navigation');
            const legend = document.getElementById('particleLegend');
            const info = document.getElementById('particleCount');
            
            // Show loading overlay
            iframe.classList.add('loading');
            loadingOverlay.classList.add('active');

            const is3D = currentMode === '3D';
            if (loadingText) loadingText.textContent = is3D ? 'Loading particle...' : 'Loading 1D model...';
            if (nav) nav.style.display = is3D ? 'flex' : 'none';
            if (legend) {
                legend.style.visibility = is3D ? 'visible' : 'hidden';
                legend.style.pointerEvents = is3D ? 'auto' : 'none';
                legend.style.opacity = is3D ? '1' : '0';
            }
            
            // Wait for fade-out transition to complete, then swap src
            setTimeout(() => {
                iframe.src = is3D ? `particle_${pid}.html` : 'model_1d_overview.html';
                
                // Send theme to new iframe after it loads
                iframe.onload = function() {
                    const isDark = !document.body.classList.contains('light-mode');
                    iframe.contentWindow.postMessage({ type: 'theme-change', isDark: isDark }, '*');
                    
                    // Fade iframe back in smoothly
                    setTimeout(() => {
                        loadingOverlay.classList.remove('active');
                        iframe.classList.remove('loading');
                    }, 100);
                };
            }, 220);

            if (is3D) {
                info.textContent = `Particle ${currentIndex + 1} of ${particles.length}`;
                document.getElementById('prevBtn').disabled = (currentIndex === 0);
                document.getElementById('nextBtn').disabled = (currentIndex === particles.length - 1);
            } else {
                info.textContent = has1DModel
                    ? '1D abundances and full-analysis outputs'
                    : '1D model is not available';
            }

            // Interactive legend behavior: selected particle highlighted with outline
            particles.forEach((particleId, idx) => {
                const chip = document.getElementById(`chip-${particleId}`);
                if (!chip) return;
                if (is3D && idx === currentIndex) {
                    chip.classList.add('active');
                } else {
                    chip.classList.remove('active');
                }
            });

            resizeFrame();
        }

        function changeParticle(direction) {
            currentIndex += direction;
            if (currentIndex < 0) currentIndex = 0;
            if (currentIndex >= particles.length) currentIndex = particles.length - 1;
            updateDisplay();
        }

        function resizeFrame() {
            const header = document.querySelector('.header');
            const nav = document.querySelector('.navigation');
            const info = document.querySelector('.info');
            const content = document.querySelector('.content');
            const frame = document.getElementById('particleFrame');
            const headerH = header ? header.offsetHeight : 0;
            const navH = nav ? nav.offsetHeight : 0;
            const infoH = info ? info.offsetHeight : 0;
            const contentStyle = window.getComputedStyle(content);
            const padTop = parseFloat(contentStyle.paddingTop || '0');
            const padBottom = parseFloat(contentStyle.paddingBottom || '0');
            const available = window.innerHeight - headerH - navH - infoH - padTop - padBottom - 6;
            frame.style.height = `${Math.max(460, available)}px`;
        }

        // Keyboard navigation
        document.addEventListener('keydown', function(event) {
            if (currentMode !== '3D') return;
            if (event.key === 'ArrowLeft') {
                changeParticle(-1);
            } else if (event.key === 'ArrowRight') {
                changeParticle(1);
            }
        });

        window.addEventListener('resize', resizeFrame);

        // Initialize
        buildLegend();
        updateChemistryButtons();
        updateModeButtons();
        updateDisplay();
    </script>
</body>
</html>
"""
    
    # Write HTML file
    html_path = output_dir / 'index.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    return html_path


def setup_inspection_directory(dir_path):
    """Always recreate inspection directory from scratch (non-interactive)."""
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    print(f"✓ Fresh inspection directory ready: {dir_path}")

    return True


def parse_args():
    parser = argparse.ArgumentParser(description='Generate interactive particle inspection interface.')
    parser.add_argument('--chemistry', default='Crich', choices=['Crich', 'Orich'])
    parser.add_argument('--pmf', default='wind_v10')
    parser.add_argument('--max-particle-id', type=int, default=300000)
    parser.add_argument('--start-index', type=int, default=2)
    parser.add_argument('--n-select', type=int, default=15)
    args, _ = parser.parse_known_args()  # ignore Jupyter/IPython injected args
    return args


def main():
    global daughters
    args = parse_args()

    to_cm = True
    savedirmain = BASE_PATH
    pmf = args.pmf
    of = 'ev_output'

    # Select particles once — same trace files for all chemistry types
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

    print(f"\nSelected particle IDs: {particle_IDs}")
    particle_IDs[1] = 29823
    particle_IDs[2] = 46371

    # Determine which chemistry types to run
    # --chemistry selects which browser tab to open; always generate both
    chemistry_types = ['Crich', 'Orich']
    preferred_open = args.chemistry  # open this one in browser at the end

    generated_paths = {}

    for chemistry_type in chemistry_types:
        daughters = daughters_Orich if chemistry_type == 'Orich' else daughters_Crich
        mf = f'evolving_model/{chemistry_type}'
        output_dir = savedirmain / 'figures' / 'particle_inspection' / chemistry_type

        print(f"\n{'=' * 60}")
        print(f"Processing {chemistry_type}")
        print(f"{'=' * 60}")

        if not setup_inspection_directory(output_dir):
            print(f"✗ Setup cancelled for {chemistry_type}, skipping")
            continue

        # Load all particle data for this chemistry type
        print(f"\nLoading {chemistry_type} particle data...")
        try:
            all_data = load_all_particles(particle_IDs, savedirmain, mf, of, to_cm)
        except FileNotFoundError as exc:
            print(f"⚠ Could not load {chemistry_type} data: {exc}")
            continue

        # Generate per-particle plots
        print(f"\nGenerating {chemistry_type} inspection plots...")
        for pid in tqdm(particle_IDs, desc=f"Creating {chemistry_type} plots"):
            data = all_data[pid]
            save_path = output_dir / f'particle_{pid}.html'
            plot_particle_inspection(pid, data, to_cm, save_path)

        # Generate 1D model overview
        has_1d_model = False
        print(f"\nGenerating {chemistry_type} 1D model overview page...")
        payload_1d = load_1d_interface_payload(savedirmain, chemistry_type=chemistry_type)
        if payload_1d is not None:
            one_d_path = output_dir / 'model_1d_overview.html'
            has_1d_model = generate_1d_overview_html(payload_1d, one_d_path, to_cm=to_cm)
            if has_1d_model:
                print(f"✓ 1D model overview created at: {one_d_path}")
        else:
            print(f"⚠ 1D model overview unavailable for {chemistry_type} (missing model files or CodeIO import)")

        # Generate HTML navigation interface
        print(f"\nGenerating {chemistry_type} HTML interface...")
        html_path = generate_html_interface(
            particle_IDs,
            output_dir,
            title=f"Particle Inspection ({chemistry_type})",
            has_1d_model=has_1d_model,
            chemistry_type=chemistry_type,
        )
        generated_paths[chemistry_type] = html_path
        print(f"✓ {chemistry_type} interface: {html_path}")

    print(f"\n✓ Complete! Generated interfaces for: {list(generated_paths.keys())}")

    # Open the preferred (or first available) chemistry type
    open_chem = (
        preferred_open
        if preferred_open in generated_paths
        else next(iter(generated_paths), None)
    )
    if open_chem:
        try:
            subprocess.run(['open', str(generated_paths[open_chem])], check=False)
            print(f"✓ Opened {open_chem} interface in browser")
        except Exception as exc:
            print(f"⚠ Could not open interface automatically: {exc}")


if __name__ == '__main__':
    main()

