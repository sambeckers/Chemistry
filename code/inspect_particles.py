import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import subprocess
from astropy import units as u
from numpy.lib import recfunctions as rfn
from convert_trace__run_models import select_particle_ids
from n_distinct_colours import generate_colormap
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import BASE_PATH


def get_distinct_colors(n_colors):
    """Get n distinct colors from n_distinct_colours, robust for small n."""
    min_safe = 14
    cmap = generate_colormap(max(n_colors, min_safe))
    return np.array(cmap.colors[:n_colors])

def load_all_particles(particle_IDs, savedirmain, mf, of, to_cm=True):
    """Load all particle trace data."""
    all_data = {}
    for pid in particle_IDs:
        filepath = savedirmain / mf / of / f"ev_{pid}.dat"
        data = np.genfromtxt(filepath, comments="#", skip_header=4, names=True)
        # Ensure data is always at least 1D (handles single-row files)
        if data.ndim == 0:
            data = np.array([data])
        x = data['X']
        y = data['Y']
        z = data['Z']
        if to_cm:
            x = (x * u.pc).to(u.cm).value
            y = (y * u.pc).to(u.cm).value
            z = (z * u.pc).to(u.cm).value
        r = np.sqrt(x**2 + y**2 + z**2)
        data = rfn.append_fields(data, 'R', r, usemask=False)
        all_data[pid] = data
    return all_data


def plot_particle_inspection(pid, data, to_cm=True, save_path=None):
    """Create interactive inspection plot for a single particle."""
    parents = ["CO", "N2", "CH4", "NH3", "H2S", "HCP", "H2O", "C2H2", "HCN", 
               "CS", "SiC2", "HCl", "HF", "C2H4", "SiO", "SiS"]
    daughters = ['CN', 'C2H', 'C4H', 'C6H', 'HC3N', 'HC5N', 'HC7N']
    
    # Divide parents into 3 groups for better visibility
    n_parent_panels = 3
    parents_per_panel = len(parents) // n_parent_panels
    parent_groups = [
        parents[:parents_per_panel],
        parents[parents_per_panel:2*parents_per_panel],
        parents[2*parents_per_panel:]
    ]
    
    fig = make_subplots(
        rows=6,
        cols=3,
        subplot_titles=[
            'Density',
            'Parents I',
            'Daughters',
            'Temperature',
            'Parents II',
            'Parents III',
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
    
    r = data['R']
    unit_label = 'cm' if to_cm else 'pc'
    
    # Left column: physical properties (no legend interaction)
    fig.add_trace(
        go.Scatter(x=r, y=data['DENSITY'], mode='lines', name='Density',
                   line=dict(color='#1f77b4', width=2), showlegend=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=r, y=data['TEMP'], mode='lines', name='Temperature',
                   line=dict(color='#d62728', width=2), showlegend=False),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=r, y=data['AV'], mode='lines', name='A_V',
                   line=dict(color='#2ca02c', width=2), showlegend=False),
        row=5, col=1
    )

    # Middle column: parent molecules (legend entries are interactive)
    colors_parents = get_distinct_colors(len(parents))
    molecule_trace_indices = []
    molecule_original_colors = []
    
    parent_positions = [(1, 2), (4, 2), (4, 3)]
    for i, parent_group in enumerate(parent_groups):
        row, col = parent_positions[i]
        for mol in parent_group:
            idx = parents.index(mol)
            color = f"rgb({int(colors_parents[idx][0]*255)}, {int(colors_parents[idx][1]*255)}, {int(colors_parents[idx][2]*255)})"
            fig.add_trace(
                go.Scatter(
                    x=r,
                    y=data[mol],
                    mode='lines',
                    name=mol,
                    line=dict(color=color, width=2),
                    showlegend=True,
                ),
                row=row,
                col=col,
            )
            molecule_trace_indices.append(len(fig.data) - 1)
            molecule_original_colors.append(color)

    # Right column: daughter molecules

    colors_daughters = get_distinct_colors(len(daughters))
    
    for i, mol in enumerate(daughters):
        color = f"rgb({int(colors_daughters[i][0]*255)}, {int(colors_daughters[i][1]*255)}, {int(colors_daughters[i][2]*255)})"
        fig.add_trace(
            go.Scatter(
                x=r,
                y=data[mol],
                mode='lines',
                name=mol,
                line=dict(color=color, width=2),
                showlegend=True,
            ),
            row=1,
            col=3,
        )
        molecule_trace_indices.append(len(fig.data) - 1)
        molecule_original_colors.append(color)

    # Axis formatting
    for row, col in [(1, 1), (3, 1), (5, 1), (1, 2), (4, 2), (1, 3), (4, 3)]:
        fig.update_xaxes(
            type='log', row=row, col=col, showgrid=True, gridcolor='rgba(0,0,0,0.2)',
            automargin=True, title_standoff=2, tickfont=dict(size=9)
        )
        fig.update_yaxes(
            type='log', row=row, col=col, showgrid=True, gridcolor='rgba(0,0,0,0.2)',
            automargin=True, title_standoff=2, tickfont=dict(size=9)
        )

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
        margin=dict(l=64, r=16, t=30, b=122),
        paper_bgcolor='#f8f9fa',
        plot_bgcolor='#ffffff',
        legend=dict(
            orientation='h',
            y=-0.17,
            yanchor='top',
            x=0.0,
            xanchor='left',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(208,215,222,1)',
            borderwidth=1,
            font=dict(size=10, color='#24292f'),
            itemwidth=40,
            itemsizing='constant'
        ),
    )
    fig.update_annotations(font=dict(size=11))
    fig.for_each_annotation(lambda ann: ann.update(yshift=4))

    # Keep axis-title fonts compact to avoid cross-panel overlap
    fig.update_xaxes(title_font=dict(size=11))
    fig.update_yaxes(title_font=dict(size=11))

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
    const moleculeTraceIndices = {molecule_trace_indices};
    const originalColors = {molecule_original_colors};
    const markerStorageKey = 'particle-markers::{pid}::' + window.location.pathname;
    let selectedTrace = -1;
    let markerTraceIndices = [];
    let focusedMarkerTrace = -1;

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
            const raw = window.localStorage.getItem(markerStorageKey);
            if (!raw) return [];
            const parsed = JSON.parse(raw);
            return Array.isArray(parsed) ? parsed : [];
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
                return {{
                    x: trace.x && trace.x.length ? trace.x[0] : null,
                    y: trace.y && trace.y.length ? trace.y[0] : null,
                    label: label,
                    color: markerColor,
                    baseTraceIndex: baseTraceIndex,
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
        const newColors = [];
        const newOpacities = [];
        for (let i = 0; i < plot.data.length; i++) {{
            const trace = plot.data[i];
            const isMolecule = moleculeTraceIndices.indexOf(i) !== -1;
            const isMarker = markerTraceIndices.indexOf(i) !== -1 || (trace && trace.meta && trace.meta.isUserMarker);
            if (isMarker) {{
                newColors.push(undefined);
                newOpacities.push(1.0);
                continue;
            }}
            if (!isMolecule) {{
                const lineColor = trace && trace.line && trace.line.color ? trace.line.color : '#1f77b4';
                newColors.push(lineColor);
                newOpacities.push(1.0);
                continue;
            }}

            const molIdx = moleculeTraceIndices.indexOf(i);
            const origColor = originalColors[molIdx];
            if (targetTrace === -1 || i === targetTrace) {{
                newColors.push(origColor);
                newOpacities.push(1.0);
            }} else {{
                newColors.push('#b8b8b8');
                newOpacities.push(0.2);
            }}
        }}
        Plotly.restyle(plot, {{'line.color': newColors, 'opacity': newOpacities}}).then(function() {{
            applyMarkerFocus();
        }});
    }}

    plot.on('plotly_legendclick', function(event) {{
        const traceIndex = event.curveNumber;
        if (moleculeTraceIndices.indexOf(traceIndex) === -1) {{
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

    plot.on('plotly_click', function(event) {{
        const isShiftClick = !!(event.event && event.event.shiftKey);
        const point = event.points && event.points.length ? event.points[0] : null;
        if (!point) return;

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

    function fitPlotToViewport() {{
        const w = Math.max(900, window.innerWidth - 4);
        const h = Math.max(520, window.innerHeight - 4);
        Plotly.relayout(plot, {{ width: w, height: h }});
    }}

    restoreMarkers();
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
        #marker-help {{
            position: fixed;
            left: 6px;
            bottom: -10px;
            z-index: 50;
            pointer-events: none;
            font-size: 12px;
            color: #57606a;
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid rgba(208, 215, 222, 0.9);
            border-radius: 8px;
            padding: 6px 10px;
        }}
        body.dark-mode #marker-help {{
            color: #d0d7de;
            background: rgba(30, 30, 30, 0.9);
            border-color: rgba(100, 100, 100, 0.7);
        }}
    </style>
</head>
<body>
{plot_div}
<div id="marker-help">Click a line to add marker. Click marker to focus/unfocus. Shift+click marker to remove.</div>
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


def generate_html_interface(particle_IDs, output_dir, title="Particle Inspection"):
    """Generate an HTML interface to navigate between particle plots."""
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
            justify-content: space-between;
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
            transition: all 0.2s ease;
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
        .navigation {{
            background-color: var(--surface);
            padding: 12px 16px;
            text-align: center;
            border-top: 1px solid var(--border);
            flex: 0 0 auto;
            z-index: 100;
        }}
        .nav-buttons {{
            display: inline-block;
            margin-right: 8px;
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
            transition: opacity 0.3s ease;
        }}
        .image-container iframe.loading {{
            opacity: 0;
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
            justify-content: center;
            align-items: center;
            gap: 6px;
            flex-wrap: wrap;
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
        <button class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">
            <span id="themeIcon">🌙</span>
        </button>
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
        <div class="nav-buttons">
            <button id="prevBtn" onclick="changeParticle(-1)">← Previous</button>
            <button id="nextBtn" onclick="changeParticle(1)">Next →</button>
        </div>
    </div>

    <script>
        const particles = """ + str(particle_IDs) + """;
        // Pastel orange palette
        const orangePalette = ['#ffb38a', '#ffc5a3', '#ffd7bc', '#ffe0c9', '#ffead6', '#ffcba4', '#ffb891', '#ffa877', '#ff9e6d', '#ffbe9f', '#ffd4b8', '#ffa060', '#ffaa70', '#ffb580', '#ffc090'];
        const legendColors = orangePalette.slice(0, particles.length);
        let currentIndex = 0;
        
        function toggleTheme() {
            document.body.classList.toggle('light-mode');
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
            
            // Show loading overlay
            iframe.classList.add('loading');
            loadingOverlay.classList.add('active');
            
            // Update src
            iframe.src = `particle_${pid}.html`;
            
            // Send theme to new iframe after it loads
            iframe.onload = function() {
                const isDark = !document.body.classList.contains('light-mode');
                iframe.contentWindow.postMessage({ type: 'theme-change', isDark: isDark }, '*');
                
                // Hide loading overlay after a short delay for smooth transition
                setTimeout(() => {
                    loadingOverlay.classList.remove('active');
                    iframe.classList.remove('loading');
                }, 150);
            };
            
            document.getElementById('particleCount').textContent = 
                `Particle ${currentIndex + 1} of ${particles.length}`;
            
            // Update button states
            document.getElementById('prevBtn').disabled = (currentIndex === 0);
            document.getElementById('nextBtn').disabled = (currentIndex === particles.length - 1);

            // Interactive legend behavior: selected particle highlighted with outline
            particles.forEach((particleId, idx) => {
                const chip = document.getElementById(`chip-${particleId}`);
                if (!chip) return;
                if (idx === currentIndex) {
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
            if (event.key === 'ArrowLeft') {
                changeParticle(-1);
            } else if (event.key === 'ArrowRight') {
                changeParticle(1);
            }
        });

        window.addEventListener('resize', resizeFrame);

        // Initialize
        buildLegend();
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


def setup_inspection_directory(base_path):
    """Create or clear inspection directory."""
    dir_path = base_path / 'figures' / 'particle_inspection'
    
    if dir_path.exists():
        if any(dir_path.iterdir()):
            print(f"\nDirectory 'particle_inspection' already exists and contains files.")
            response = input(f"Empty 'particle_inspection' directory? [y/n]: ").strip().lower()
            
            if response in {'y', 'yes'}:
                print(f"   Removing all files in particle_inspection/...")
                shutil.rmtree(dir_path)
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"   ✓ particle_inspection/ cleared")
            else:
                print(f"✗ Keeping existing files in particle_inspection/")
                return False
        else:
            print(f"✓ particle_inspection/ exists (empty)")
    else:
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created particle_inspection/")
    
    return True


def main():
    to_cm = True
    savedirmain = BASE_PATH
    pmf = 'wind_v10'
    mf = 'evolving_model'
    of = 'ev_output'
    
    # Setup output directory
    output_dir = savedirmain / 'figures' / 'particle_inspection'
    if not setup_inspection_directory(savedirmain):
        print("\n✗ Setup cancelled by user or directory not empty")
        return
    
    # Select particles
    particle_IDs_file = savedirmain / 'traces' / pmf / 'particle_IDs.txt'
    tracesf = f'traces/{pmf}/trace_output_with_av'
    trace_dir = savedirmain / tracesf
    particle_IDs = select_particle_ids(
        particle_IDs_file,
        trace_dir,
        max_particle_id=300000,
        start_index=2,
        n_select=15,
    )
    
    print(f"\nSelected particle IDs: {particle_IDs}")
    particle_IDs[1] = 29823
    particle_IDs[2] = 46371
    
    # Load all particle data
    print("\nLoading particle data...")
    all_data = load_all_particles(particle_IDs, savedirmain, mf, of, to_cm)
    
    # Generate plots for each particle
    print("\nGenerating inspection plots...")
    for pid in tqdm(particle_IDs, desc="Creating plots"):
        data = all_data[pid]
        save_path = output_dir / f'particle_{pid}.html'
        plot_particle_inspection(pid, data, to_cm, save_path)
    
    # Generate HTML interface
    print("\nGenerating HTML interface...")
    html_path = generate_html_interface(particle_IDs, output_dir, 
                                       title="Particle Inspection Interface")
    
    print(f"\n✓ Complete!")
    print(f"✓ Generated {len(particle_IDs)} particle inspection plots")
    print(f"✓ HTML interface created at: {html_path}")
    print(f"\nOpen the interface by running:")
    print(f"  open {html_path}")

    try:
        subprocess.run(['open', str(html_path)], check=False)
        print(f"✓ Opened interface in browser")
    except Exception as exc:
        print(f"⚠ Could not open interface automatically: {exc}")


if __name__ == '__main__':
    main()

