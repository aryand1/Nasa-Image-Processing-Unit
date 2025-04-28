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

# ── 1) Model & colors ───────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\aryand\Downloads\weights_DP(6).pt"
model      = YOLO(MODEL_PATH)

# segmentation mask colors
COLOR_MAP = {
    "water": (255,   0,   0),   # red
    "ice":   (  0, 128,   0)    # green
}

# overlap highlight colors
OVERLAP_COLORS = {
    "ww": (  0,   0, 255),   # blue
    "ii": (255, 165,   0),   # orange
    "wi": (255, 255,   0)    # yellow
}

ALPHA_SEG     = 0.5   # opacity for water/ice masks
ALPHA_OVERLAP = 0.65  # opacity for overlap highlights

def blend_mask(base_img, mask, color, alpha):
    """Blend `color` into `base_img` where `mask==1`, at opacity `alpha`."""
    idx = mask.astype(bool)
    base_img[idx] = base_img[idx] * (1 - alpha) + np.array(color) * alpha
    return base_img

def apply_full_overlay(img, masks_np, class_names, mask_thresh=0.3):
    """
    Build and blend:
      - water mask (red @50%)
      - ice mask   (green@50%)
      - water–water overlap (blue @65%)
      - ice–ice overlap   (orange@65%)
      - water–ice overlap  (yellow@65%)
    """
    h, w = img.shape[:2]

    # 1) full-size binaries
    full_masks = [
        cv2.resize((m > mask_thresh).astype(np.uint8), (w, h),
                   interpolation=cv2.INTER_NEAREST)
        for m in masks_np
    ]

    # 2) union water & ice
    water_union = np.zeros((h, w), dtype=np.uint8)
    ice_union   = np.zeros((h, w), dtype=np.uint8)
    for fm, cls in zip(full_masks, class_names):
        if cls == "water":
            water_union |= fm
        elif cls == "ice":
            ice_union   |= fm

    # 3) compute overlaps
    ww_mask = np.zeros((h, w), dtype=np.uint8)
    ii_mask = np.zeros((h, w), dtype=np.uint8)
    wi_mask = np.zeros((h, w), dtype=np.uint8)
    n = len(full_masks)
    for i in range(n):
        for j in range(i+1, n):
            inter = full_masks[i] & full_masks[j]
            if not np.any(inter):
                continue
            ni, nj = class_names[i], class_names[j]
            if ni == nj == "water":
                ww_mask |= inter
            elif ni == nj == "ice":
                ii_mask |= inter
            else:
                wi_mask |= inter

    # 4) blend everything
    base = img.astype(float)
    base = blend_mask(base, water_union, COLOR_MAP["water"], ALPHA_SEG)
    base = blend_mask(base, ice_union,   COLOR_MAP["ice"],   ALPHA_SEG)
    base = blend_mask(base, ww_mask,     OVERLAP_COLORS["ww"], ALPHA_OVERLAP)
    base = blend_mask(base, ii_mask,     OVERLAP_COLORS["ii"], ALPHA_OVERLAP)
    base = blend_mask(base, wi_mask,     OVERLAP_COLORS["wi"], ALPHA_OVERLAP)

    return np.clip(base, 0, 255).astype(np.uint8)

# ── 2) Dash setup ───────────────────────────────────────────────────────────
app    = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# top‐level layout: router + stores
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="result-store",  storage_type="memory"),
    dcc.Store(id="overlap-store", storage_type="memory"),
    html.Div(id="page-content")
])

# ── 3) Page 1 ───────────────────────────────────────────────────────────────
def page_1_layout():
    return html.Div(
        style={
            "height":"100vh","width":"100vw","display":"flex","flexDirection":"row",
            "background":"linear-gradient(90deg,#000,#737373)","color":"white",
            "overflow":"hidden","boxSizing":"border-box","margin":0,"padding":0
        },
        children=[

            # left column
            html.Div(
                style={
                    "flex":"0 0 35%","display":"flex","flexDirection":"column",
                    "justifyContent":"space-between","textAlign":"center",
                    "height":"100%","padding":"1rem","boxSizing":"border-box"
                },
                children=[
                    # title
                    html.Div([
                        html.H1("NASA", style={"textDecoration":"underline","fontSize":"45px"}),
                        html.H2("Image Processing Unit", style={"textDecoration":"underline","fontSize":"45px"})
                    ]),
                    # folder input
                    html.Div([
                        html.Button(
                            html.Img(src="/assets/cloud_upload1.png", style={"width":"150px","cursor":"pointer"}),
                            id="folder-picker-btn", n_clicks=0,
                            style={"border":"none","background":"transparent"}
                        ),
                        dcc.Input(
                            id="folder-input",
                            placeholder="Paste folder path here",
                            type="text",
                            style={"width":"100%","marginTop":"1rem","fontSize":"1.25rem"}
                        )
                    ]),
                    # toggle / spinner / run / download
                    html.Div([
                        html.Div(
                            style={"display":"flex","alignItems":"center","marginTop":"0.5rem"},
                            children=[
                                daq.ToggleSwitch(
                                    id="overlay-toggle", value=True, size=55, color="red"
                                ),
                                html.Span("Save segmentation overlays",
                                          style={"marginLeft":"10px","fontSize":"22px"})
                            ]
                        ),
                        html.Div(
                            dcc.Loading(
                                html.Div(id="progress-text", style={"fontSize":"1.25rem"}),
                                type="circle"
                            ),
                            style={"display":"flex","justifyContent":"center","margin":"1rem 0"}
                        ),
                        html.Div(
                            style={"display":"flex","justifyContent":"center"},
                            children=[
                                html.Button(
                                    "Run Detection", id="run-btn", n_clicks=0,
                                    style={"fontSize":"1.25rem","padding":"0.75rem 1.5rem"}
                                )
                            ]
                        ),
                        dcc.Download(id="download-summary")
                    ], id="unit3")
                ]
            ),

            # right column: preview image
            html.Div(
                html.Img(src="/assets/Ice_image.jpg",
                         style={"width":"100%","height":"100%","objectFit":"cover","display":"block"}),
                style={"flex":"1","height":"100%","overflow":"hidden"}
            )
        ]
    )

# ── 4) Page 2 ───────────────────────────────────────────────────────────────
def page_2_layout():
    return html.Div(
        style={
            "position":"relative","height":"100vh","width":"100vw",
            "background":"linear-gradient(90deg,#000,#737373)","color":"white",
            "overflowY":"auto","margin":0,"padding":0,"boxSizing":"border-box"
        },
        children=[
            # back button
            dcc.Link(
                html.Button("Back to Home",
                            style={"position":"absolute","top":"1rem","left":"1rem",
                                   "backgroundColor":"#e6c645","color":"white","border":"none",
                                   "padding":"0.5rem 1rem","borderRadius":"12px","cursor":"pointer","opacity":"0.8"}),
                href="/"
            ),

            # two combined graphs side by side
            html.Div(
                style={"display":"flex","justifyContent":"space-between","width":"95%","margin":"4rem auto"},
                children=[
                    html.Div([
                        html.H4("Water (%) & Ice (%)", style={"textAlign":"center"}),
                        dcc.Graph(id="pct-graph"),
                        html.Ul([
                            html.Li("Blue = Water (%)"),
                            html.Li("Red = Ice (%)")
                        ], style={"listStyle":"disc","paddingLeft":"1.5rem","marginTop":"0.5rem"})
                    ], style={"width":"48%"}),

                    html.Div([
                        html.H4("Overlap Counts", style={"textAlign":"center"}),
                        dcc.Graph(id="ov-graph"),
                        html.Ul([
                            html.Li("Blue = Water–Water"),
                            html.Li("Red  = Ice–Ice"),
                            html.Li("Green= Water–Ice")
                        ], style={"listStyle":"disc","paddingLeft":"1.5rem","marginTop":"0.5rem"})
                    ], style={"width":"48%"})
                ]
            ),

            # slider + text
            html.Div(id="slider-output-container", style={"marginTop":"20px","textAlign":"center"}),
            html.Div(
                dcc.Slider(
                    id="entry-slider", min=1, max=1, value=1, step=1, marks=None,
                    tooltip={"placement":"bottom","always_visible":True}, updatemode="drag"
                ),
                style={"width":"80%","margin":"0 auto","paddingBottom":"30px"}
            ),

            # donuts
            html.Div(
                style={"width":"95%","margin":"2rem auto","textAlign":"center"},
                children=[
                    html.Div([
                        dcc.Graph(id="donut-water"),
                        html.Div("Water Count", style={"fontWeight":"bold"})
                    ], style={"width":"25%","display":"inline-block"}),
                    html.Div([
                        dcc.Graph(id="donut-ice"),
                        html.Div("Ice Count", style={"fontWeight":"bold"})
                    ], style={"width":"25%","display":"inline-block"}),
                    html.Div([
                        dcc.Graph(id="donut-void"),
                        html.Div("Void Percentage", style={"fontWeight":"bold"})
                    ], style={"width":"25%","display":"inline-block"}),
                    html.Div([
                        dcc.Graph(id="donut-conf"),
                        html.Div("Avg Confidence", style={"fontWeight":"bold"})
                    ], style={"width":"25%","display":"inline-block"})
                ]
            ),

            # overall overlap summary
            html.Div(id="overlap-summary",
                     style={"width":"95%","margin":"2rem auto","fontSize":"18px"})
        ]
    )

# ── 5) Router ───────────────────────────────────────────────────────────────
@app.callback(
    Output("page-content","children"),
    Input("url","pathname")
)
def display_page(pathname):
    return page_2_layout() if pathname=="/summary" else page_1_layout()

# ── 6) Clientside folder picker ─────────────────────────────────────────────
app.clientside_callback(
    """
    function(n_clicks) {
      if (!n_clicks) return "";
      const f = window.prompt("Enter folder path:");
      return f || "";
    }
    """,
    Output("folder-input","value"),
    Input("folder-picker-btn","n_clicks")
)

# ── 7) run_detection ───────────────────────────────────────────────────────
@app.callback(
    Output("progress-text","children"),
    Output("download-summary","data"),
    Output("result-store","data"),
    Output("overlap-store","data"),
    Output("url","pathname"),
    Input("run-btn","n_clicks"),
    State("folder-input","value"),
    State("overlay-toggle","value"),
    prevent_initial_call=True
)
def run_detection(n_clicks, folder, save_ovl):
    if not n_clicks:
        raise PreventUpdate
    if not folder or not os.path.isdir(folder):
        return "❌ Invalid folder path", None, no_update, no_update, no_update

    seg_dir = os.path.join(folder, "segmentation results")
    if save_ovl and not os.path.exists(seg_dir):
        os.makedirs(seg_dir)

    rows = []
    overlap_totals = {"ww":0,"ii":0,"mixed":0}

    for root, dirs, files in os.walk(folder):
        if save_ovl and "segmentation results" in dirs:
            dirs.remove("segmentation results")
        for fname in files:
            ext = os.path.splitext(fname.lower())[1]
            if ext not in {".png",".jpg",".jpeg",".tif",".tiff"}:
                continue

            path = os.path.join(root, fname)
            bgr  = cv2.imread(path)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            res = model(img)[0]

            h, w = res.orig_shape
            total_px = w*h
            water_cnt=ice_cnt=0
            water_area=ice_area=0
            confs = []

            # per-image overlap counts
            ww_count=ii_count=wi_count=0

            if res.masks is not None and len(res.boxes):
                masks_np = res.masks.data.cpu().numpy()
                boxes    = list(res.boxes)
                class_names = [res.names[int(b.cls)].lower() for b in boxes]
                bin_masks   = [(m>0.3).astype(np.uint8) for m in masks_np]

                # count & conf
                for fm, box in zip(bin_masks, boxes):
                    full_m = cv2.resize(fm, (w,h), interpolation=cv2.INTER_NEAREST)
                    area = full_m.sum()
                    cls_name = res.names[int(box.cls)].lower()
                    if cls_name=="water":
                        water_cnt  +=1; water_area+=area
                    elif cls_name=="ice":
                        ice_cnt    +=1; ice_area  +=area
                    confs.append(box.conf.item())

                water_pct = (water_area/total_px*100) if total_px else 0
                ice_pct   = (ice_area/total_px*100)   if total_px else 0
                void_pct  = max(0,100-water_pct-ice_pct)
                avg_conf  = (sum(confs)/len(confs)*100) if confs else 0

                # compute overlaps
                full_masks = [cv2.resize(m, (w,h), interpolation=cv2.INTER_NEAREST)
                              for m in bin_masks]
                for i in range(len(full_masks)):
                    for j in range(i+1, len(full_masks)):
                        inter = full_masks[i] & full_masks[j]
                        if not np.any(inter):
                            continue
                        ni, nj = class_names[i], class_names[j]
                        if ni==nj=="water":
                            overlap_totals["ww"]+=1; ww_count+=1
                        elif ni==nj=="ice":
                            overlap_totals["ii"]+=1; ii_count+=1
                        else:
                            overlap_totals["mixed"]+=1; wi_count+=1

                if save_ovl:
                    over = apply_full_overlay(img, masks_np, class_names)
                    out_fn = os.path.splitext(fname)[0] + "_fullovl.png"
                    cv2.imwrite(os.path.join(seg_dir,out_fn),
                                cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
            else:
                water_pct=ice_pct=void_pct=avg_conf=0

            rows.append({
                "Image Name":           fname,
                "water_cnt":            water_cnt,
                "ice_cnt":              ice_cnt,
                "void_pct":             void_pct,
                "avg_conf":             avg_conf,
                "Overlap_Water-Water":  ww_count,
                "Overlap_Ice-Ice":      ii_count,
                "Overlap_Water-Ice":    wi_count,
                "Water (%)":            round(water_pct,2),
                "Ice (%)":              round(ice_pct,2),
                "Avg Confidence (%)":   round(avg_conf,2)
            })

    # write Excel
    df = pd.DataFrame(rows)
    path = os.path.join(folder,"detection_summary.xlsx")
    df.to_excel(path, index=False)

    return (
        "✅ Processing complete!",
        dcc.send_file(path),
        rows,
        overlap_totals,
        "/summary"
    )

# ── 8) slider & graphs ─────────────────────────────────────────────────────
@app.callback(
    Output("pct-graph","figure"),
    Output("ov-graph","figure"),
    Output("slider-output-container","children"),
    Input("entry-slider","value"),
    State("result-store","data")
)
def update_graphs(val, rows):
    if not rows:
        return {},{}, "No data."
    df = pd.DataFrame(rows)
    val = min(val, len(df))
    dff = df.iloc[:val]

    fig1 = px.line(dff, x=dff.index, y=["Water (%)","Ice (%)"],
                   labels={"index":"Entry","value":"%","variable":"Metric"},
                   markers=True, line_shape="spline")
    fig1.update_layout(title=f"Water & Ice (%) (first {val})",
                       margin=dict(l=40,r=10,t=30,b=30),
                       transition={'duration':500,'easing':'cubic-in-out'})

    fig2 = px.line(dff, x=dff.index,
                   y=["Overlap_Water-Water","Overlap_Ice-Ice","Overlap_Water-Ice"],
                   labels={"index":"Entry","value":"Count","variable":"Metric"},
                   markers=True, line_shape="spline")
    fig2.update_layout(title=f"Overlaps (first {val})",
                       margin=dict(l=40,r=10,t=30,b=30),
                       transition={'duration':500,'easing':'cubic-in-out'})

    return fig1, fig2, f"Showing first {val} of {len(df)}"

# ── 9) slider max/value ────────────────────────────────────────────────────
@app.callback(
    Output("entry-slider","max"),
    Output("entry-slider","value"),
    Input("result-store","data")
)
def update_slider(rows):
    if not rows:
        raise PreventUpdate
    n = len(rows)
    return n, n

# ── 10) donuts ─────────────────────────────────────────────────────────────
@app.callback(
    Output("donut-water","figure"),
    Output("donut-ice","figure"),
    Output("donut-void","figure"),
    Output("donut-conf","figure"),
    Input("result-store","data")
)
def update_donuts(rows):
    if not rows:
        return {},{}, {},{}
    df = pd.DataFrame(rows)

    fig_w = px.pie(names=["Water"], values=[df["water_cnt"].max()], hole=0.6)
    fig_w.update_traces(textinfo="value", textposition="inside")
    fig_w.update_layout(showlegend=False, margin=dict(l=20,r=20,t=30,b=20))

    fig_i = px.pie(names=["Ice"], values=[df["ice_cnt"].max()], hole=0.6)
    fig_i.update_traces(textinfo="value", textposition="inside")
    fig_i.update_layout(showlegend=False, margin=dict(l=20,r=20,t=30,b=20))

    fig_v = px.pie(names=["Void",""], values=[df["void_pct"].max(), 100-df["void_pct"].max()], hole=0.6)
    fig_v.update_traces(textinfo="percent", textposition="inside")
    fig_v.update_layout(showlegend=False, margin=dict(l=20,r=20,t=30,b=20))

    fig_c = px.pie(names=["Conf",""], values=[df["avg_conf"].mean(), 100-df["avg_conf"].mean()], hole=0.6)
    fig_c.update_traces(textinfo="percent", textposition="inside")
    fig_c.update_layout(showlegend=False, margin=dict(l=20,r=20,t=30,b=20))

    return fig_w, fig_i, fig_v, fig_c

# ── 11) overlap summary ────────────────────────────────────────────────────
@app.callback(
    Output("overlap-summary","children"),
    Input("overlap-store","data")
)
def display_overlap(o):
    if not o:
        return html.P("No overlap data.")
    return html.Div([
        html.H4("Overlap Summary", style={"textAlign":"center"}),
        html.Ul([
            html.Li(f"Water–Water total: {o['ww']}"),
            html.Li(f"Ice–Ice total:   {o['ii']}"),
            html.Li(f"Water–Ice total: {o['mixed']}")
        ], style={"listStyle":"none","padding":0})
    ])

if __name__ == "__main__":
    app.run(debug=True)
