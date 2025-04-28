import os
import cv2
import numpy as np
import pandas as pd
import plotly.express as px

import dash
from dash import html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import dash_daq as daq
from ultralytics import YOLO

# ── 1) Load your model & set up color map ────────────────────────────────────
MODEL_PATH = r"C:\Users\aryand\Downloads\weights_DP(6).pt"
model      = YOLO(MODEL_PATH)
COLOR_MAP  = {"water": (255, 0, 0), "ice": (0, 128, 0)}

def apply_segmentation_mask(image, results, color_map, mask_thresh=0.3):
    """Overlay segmentation masks on an image."""
    overlay = image.copy()
    h, w    = image.shape[:2]
    if results.masks is None:
        return image
    for mask, cls in zip(
        results.masks.data.cpu().numpy(),
        results.boxes.cls.cpu().numpy()
    ):
        name   = results.names[int(cls)].lower()
        color  = color_map.get(name, (0, 0, 0))
        bin_m  = (mask > mask_thresh).astype(np.uint8)
        full_m = cv2.resize(bin_m, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay[full_m == 1] = color
    return cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

# ── 2) Initialize Dash ───────────────────────────────────────────────────────
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="result-store",  storage_type="memory"),
    dcc.Store(id="overlap-store", storage_type="memory"),
    html.Div(id="page-content")
])

# ── 3) Page 1 layout ────────────────────────────────────────────────────────
def page_1_layout():
    return html.Div(
        style={
            "height": "100vh", "width": "100vw",
            "display": "flex", "flexDirection": "row",
            "background": "linear-gradient(90deg, #000000, #737373)",
            "color": "white", "overflow": "hidden",
            "boxSizing": "border-box", "margin": 0, "padding": 0
        },
        children=[

            # Left column
            html.Div(
                style={
                    "flex": "0 0 35%", "display": "flex", "flexDirection": "column",
                    "justifyContent": "space-between", "textAlign": "center",
                    "height": "100%", "padding": "1rem", "boxSizing": "border-box",
                    "overflow": "hidden"
                },
                children=[

                    # Title
                    html.Div([
                        html.H1("NASA", style={"textDecoration": "underline", "fontSize": "45px"}),
                        html.H2("Image Processing Unit", style={"textDecoration": "underline", "fontSize": "45px"})
                    ]),

                    # Cloud icon + folder input
                    html.Div([
                        html.Button(
                            html.Img(src="/assets/cloud_upload1.png",
                                     style={"width": "150px", "cursor": "pointer"}),
                            id="folder-picker-btn", n_clicks=0,
                            style={"border": "none", "background": "transparent"}
                        ),
                        dcc.Input(
                            id="folder-input",
                            placeholder="Paste folder path here",
                            type="text",
                            style={"width": "100%", "marginTop": "1rem", "fontSize": "1.25rem"}
                        )
                    ]),

                    # Toggle + Run, spinner + progress, download
                    html.Div([
                        # row: toggle + button
                        html.Div(
                            style={
                                "display": "flex", "justifyContent": "space-between",
                                "alignItems": "center", "marginTop": "0.5rem"
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex", "alignItems": "center"},
                                    children=[
                                        daq.ToggleSwitch(
                                            id="overlay-toggle",
                                            value=True,
                                            size=55,
                                            color="red"
                                        ),
                                        html.Span("Save segmentation overlays",
                                                  style={"marginLeft": "10px", "fontSize": "22px"})
                                    ]
                                ),
                                html.Button(
                                    "Run Detection",
                                    id="run-btn",
                                    n_clicks=0,
                                    style={"fontSize": "1.25rem", "padding": "0.75rem 1.5rem"}
                                )
                            ]
                        ),
                        # spinner + text
                        dcc.Loading(
                            html.Div(id="progress-text",
                                     style={"marginTop": "1rem", "fontSize": "1.25rem"}),
                            type="circle"
                        ),
                        # hidden download
                        dcc.Download(id="download-summary")
                    ], id="unit3")
                ]
            ),

            # Right column: image preview
            html.Div(
                html.Img(
                    src="/assets/Ice_image.jpg",
                    style={"width": "100%", "height": "100%", "objectFit": "cover", "display": "block"}
                ),
                style={"flex": "1", "height": "100%", "overflow": "hidden"}
            )
        ]
    )

# ── 4) Page 2 layout ────────────────────────────────────────────────────────
def page_2_layout():
    return html.Div(
        style={
            "position": "relative", "height": "100vh", "width": "100vw",
            "background": "linear-gradient(90deg, #000000, #737373)",
            "color": "white", "overflowY": "auto",
            "margin": 0, "padding": 0, "boxSizing": "border-box"
        },
        children=[
            # Back to home button
            dcc.Link(
                html.Button(
                    "Back to Home",
                    className="back-link",
                    style={
                        "position": "absolute", "top": "1rem", "left": "1rem",
                        "backgroundColor": "#e6c645", "color": "white", "border": "none",
                        "padding": "0.5rem 1rem", "borderRadius": "12px",
                        "cursor": "pointer", "opacity": "0.8", "transition": "opacity 0.2s"
                    }
                ),
                href="/"
            ),

            # Graphs side by side
            html.Div([
                html.Div([
                    html.H4("Water (%) Trend", style={"textAlign": "center"}),
                    dcc.Graph(id="water-graph")
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "padding": "0 1%"}),
                html.Div([
                    html.H4("Ice (%) Trend", style={"textAlign": "center"}),
                    dcc.Graph(id="ice-graph")
                ], style={"width": "48%", "display": "inline-block", "verticalAlign": "top", "padding": "0 1%"})
            ], style={"width": "95%", "margin": "4rem auto", "textAlign": "center"}),

            # Slider + text
            html.Div(id="slider-output-container", style={"marginTop": "20px", "textAlign": "center"}),
            html.Div(
                dcc.Slider(
                    id="entry-slider",
                    min=1, max=1, value=1, step=1, marks=None,
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode="drag"
                ),
                style={"width": "80%", "margin": "0 auto", "paddingBottom": "30px"}
            ),

            # Donut charts
            html.Div([
                html.Div(dcc.Graph(id="donut-water"), style={"width": "25%", "display": "inline-block"}),
                html.Div(dcc.Graph(id="donut-ice"  ), style={"width": "25%", "display": "inline-block"}),
                html.Div(dcc.Graph(id="donut-void" ), style={"width": "25%", "display": "inline-block"}),
                html.Div(dcc.Graph(id="donut-conf" ), style={"width": "25%", "display": "inline-block"}),
            ], style={"width": "95%", "margin": "2rem auto", "textAlign": "center"}),

            # Overlap summary
            html.Div(id="overlap-summary",
                     style={"width": "95%", "margin": "2rem auto", "color": "white", "fontSize": "18px"})
        ]
    )

# ── 5) Router callback ─────────────────────────────────────────────────────
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    return page_2_layout() if pathname == "/summary" else page_1_layout()

# ── 6) Clientside prompt for folder ───────────────────────────────────────
app.clientside_callback(
    """
    function(n_clicks) {
      if (!n_clicks) return "";
      const f = window.prompt("Enter folder path:");
      return f || "";
    }
    """,
    Output("folder-input", "value"),
    Input("folder-picker-btn", "n_clicks")
)

# ── 7a) run_detection: save overlays, write Excel, download, store & nav ──
@app.callback(
    Output("progress-text",     "children"),
    Output("download-summary",  "data"),
    Output("result-store",      "data"),
    Output("overlap-store",     "data"),
    Output("url",               "pathname"),
    Input("run-btn",            "n_clicks"),
    State("folder-input",       "value"),
    State("overlay-toggle",     "value"),
    prevent_initial_call=True
)
def run_detection(n_clicks, folder, save_ovl):
    if not n_clicks:
        raise PreventUpdate
    if not folder or not os.path.isdir(folder):
        return "❌ Invalid folder path", None, no_update, no_update, no_update

    # Prepare overlay directory
    seg_dir = os.path.join(folder, "segmentation results")
    if save_ovl and not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    rows = []
    overlap_totals = {"ww": 0, "ii": 0, "mixed": 0}

    for root, dirs, files in os.walk(folder, topdown=True):
        if save_ovl and "segmentation results" in dirs:
            dirs.remove("segmentation results")
        for fname in files:
            ext = os.path.splitext(fname.lower())[1]
            if ext not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"} or \
               (save_ovl and fname.endswith("_ovl.png")):
                continue

            path = os.path.join(root, fname)
            bgr  = cv2.imread(path)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = model(img)[0]

            h, w        = res.orig_shape
            total_px    = w * h
            water_cnt   = ice_cnt = 0
            water_area  = ice_area = 0
            confs       = []

            has_ww = has_ii = has_mixed = 0

            if res.boxes is not None and len(res.boxes):
                masks_np   = res.masks.data.cpu().numpy()
                classes_np = res.boxes.cls.cpu().numpy()
                bin_masks  = [(m > 0.3).astype(np.uint8) for m in masks_np]

                # Compute counts & areas
                for mask, box in zip(masks_np, res.boxes):
                    cls_name = res.names[int(box.cls)].lower()
                    confs.append(box.conf.item())
                    bin_m  = (mask > 0.3).astype(np.uint8)
                    full_m = cv2.resize(bin_m, (w, h), interpolation=cv2.INTER_NEAREST)
                    area   = full_m.sum()
                    if cls_name == "water":
                        water_cnt  += 1; water_area += area
                    elif cls_name == "ice":
                        ice_cnt    += 1; ice_area   += area

                water_pct = (water_area / total_px * 100) if total_px else 0
                ice_pct   = (ice_area   / total_px * 100) if total_px else 0
                void_pct  = max(0, 100 - water_pct - ice_pct)
                avg_conf  = (sum(confs) / len(confs) * 100) if confs else 0

                # Detect overlaps
                n = len(bin_masks)
                for i in range(n):
                    for j in range(i+1, n):
                        if np.any(bin_masks[i] & bin_masks[j]):
                            name_i = res.names[int(classes_np[i])].lower()
                            name_j = res.names[int(classes_np[j])].lower()
                            if name_i == name_j == "water":
                                overlap_totals["ww"] += 1; has_ww = 1
                            elif name_i == name_j == "ice":
                                overlap_totals["ii"] += 1; has_ii = 1
                            else:
                                overlap_totals["mixed"] += 1; has_mixed = 1
            else:
                water_pct = ice_pct = void_pct = avg_conf = 0

            # Save overlay if requested
            if save_ovl:
                ovl     = apply_segmentation_mask(img, res, COLOR_MAP)
                bgr_ovl = cv2.cvtColor(ovl, cv2.COLOR_RGB2BGR)
                out_fn  = os.path.splitext(fname)[0] + "_ovl.png"
                cv2.imwrite(os.path.join(seg_dir, out_fn), bgr_ovl)

            rows.append({
                "Image Name":            fname,
                "water_cnt":             water_cnt,
                "ice_cnt":               ice_cnt,
                "void_pct":              void_pct,
                "avg_conf":              avg_conf,
                "Overlap_Water-Water":   has_ww,
                "Overlap_Ice-Ice":       has_ii,
                "Overlap_Water-Ice":     has_mixed,
                "Water (%)":             round(water_pct,    2),
                "Ice (%)":               round(ice_pct,      2),
                "Avg Confidence (%)":    round(avg_conf,     2)
            })

    # Write summary Excel
    df           = pd.DataFrame(rows)
    summary_path = os.path.join(folder, "detection_summary.xlsx")
    df.to_excel(summary_path, index=False)

    return (
        "✅ Processing complete!",
        dcc.send_file(summary_path),
        rows,
        overlap_totals,
        "/summary"
    )

# ── 7b) Toggle color update ───────────────────────────────────────────────
@app.callback(
    Output("overlay-toggle", "color"),
    Input("overlay-toggle", "value"),
    prevent_initial_call=False
)
def update_toggle_color(is_on):
    return "green" if is_on else "red"

# ── 8a) Update slider max/value ──────────────────────────────────────────
@app.callback(
    Output("entry-slider", "max"),
    Output("entry-slider", "value"),
    Input("result-store", "data")
)
def update_slider(rows):
    if not rows:
        raise PreventUpdate
    n = len(rows)
    return n, n

# ── 8b) Update line graphs ───────────────────────────────────────────────
@app.callback(
    Output("water-graph",             "figure"),
    Output("ice-graph",               "figure"),
    Output("slider-output-container","children"),
    Input("entry-slider",            "value"),
    State("result-store",            "data")
)
def update_graphs(slider_value, rows):
    if not rows:
        return {}, {}, "No data to display."
    df  = pd.DataFrame(rows)
    val = min(slider_value, len(df))
    dff = df.iloc[:val]

    fig_w = px.line(
        dff, x=dff.index, y="Water (%)",
        title=f"Water (%) Trend (first {val})",
        labels={"index":"Entry","Water (%)":"Water (%)"},
        markers=True, line_shape='spline'
    )
    fig_w.update_layout(
        margin=dict(l=40, r=10, t=30, b=30),
        transition={'duration': 500, 'easing': 'cubic-in-out'}
    )

    fig_i = px.line(
        dff, x=dff.index, y="Ice (%)",
        title=f"Ice (%) Trend (first {val})",
        labels={"index":"Entry","Ice (%)":"Ice (%)"},
        markers=True, line_shape='spline'
    )
    fig_i.update_layout(
        margin=dict(l=40, r=10, t=30, b=30),
        transition={'duration': 500, 'easing': 'cubic-in-out'}
    )

    return fig_w, fig_i, f"Showing first {val} out of {len(df)} entries."

# ── 9) Donut charts callback ───────────────────────────────────────────────
@app.callback(
    Output("donut-water", "figure"),
    Output("donut-ice",   "figure"),
    Output("donut-void",  "figure"),
    Output("donut-conf",  "figure"),
    Input("result-store", "data")
)
def update_donuts(rows):
    if not rows:
        return {}, {}, {}, {}
    df = pd.DataFrame(rows)

    # raw‐count donuts
    max_w = df["water_cnt"].max()
    fig_w = px.pie(names=["Water"], values=[max_w], hole=0.6)
    fig_w.update_traces(textinfo="value", textposition="inside")
    fig_w.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))

    max_i = df["ice_cnt"].max()
    fig_i = px.pie(names=["Ice"], values=[max_i], hole=0.6)
    fig_i.update_traces(textinfo="value", textposition="inside")
    fig_i.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))

    # percent donuts
    max_v = df["void_pct"].max()
    fig_v = px.pie(names=["Void", ""], values=[max_v, 100 - max_v], hole=0.6)
    fig_v.update_traces(textinfo="percent", textposition="inside")
    fig_v.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))

    avg_c = df["avg_conf"].mean()
    fig_c = px.pie(names=["Conf", ""], values=[avg_c, 100 - avg_c], hole=0.6)
    fig_c.update_traces(textinfo="percent", textposition="inside")
    fig_c.update_layout(showlegend=False, margin=dict(l=20, r=20, t=30, b=20))

    return fig_w, fig_i, fig_v, fig_c

# ── 10) Overlap summary callback ───────────────────────────────────────────
@app.callback(
    Output("overlap-summary", "children"),
    Input("overlap-store", "data")
)
def display_overlap(o):
    if not o:
        return html.P("No overlap data.", style={"color": "white"})
    return html.Div([
        html.H4("Overlap Summary", style={"textAlign": "center", "marginBottom": "0.5rem"}),
        html.Ul([
            html.Li(f"Water–Water overlaps: {o['ww']}"),
            html.Li(f"Ice–Ice overlaps:   {o['ii']}"),
            html.Li(f"Water–Ice overlaps:  {o['mixed']}")
        ], style={"listStyle": "none", "padding": 0, "color": "white"})
    ])

if __name__ == "__main__":
    app.run(debug=True)
