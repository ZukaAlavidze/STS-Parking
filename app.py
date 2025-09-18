
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pyproj import Transformer

st.set_page_config(page_title="Tbilisi Parking Map", layout="wide")

st.title("🅿️ Tbilisi Parking — Interactive Map")

@st.cache_data
def load_data(f):
    if isinstance(f, str):
        df = pd.read_csv(f)
    else:
        df = pd.read_csv(f)
    # Normalize column names (strip and unify)
    df.columns = [c.strip() for c in df.columns]
    return df

def convert_xy_to_latlon(df, x_col="X", y_col="Y"):
    # The template appears to be in UTM Zone 38N (EPSG:32638). Convert to WGS84 (EPSG:4326)
    transformer = Transformer.from_crs(32638, 4326, always_xy=True)
    xs = df[x_col].astype(float)
    ys = df[y_col].astype(float)
    lon, lat = transformer.transform(xs.values, ys.values)
    out = df.copy()
    out["lon"] = lon
    out["lat"] = lat
    return out

def coalesce_numeric(df, col):
    return pd.to_numeric(df.get(col), errors="coerce").fillna(0)

def prepare_data(df):
    # Column name hints from template
    # Required columns
    required = ["X", "Y", "Name", "Cadastral Code", "Address", "Property Owner"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.warning(f"Missing expected columns: {missing}. Please ensure your CSV matches the template.")
    # Convert coords
    df = df.dropna(subset=["X","Y"])
    df = convert_xy_to_latlon(df, "X", "Y")
    # Parking spot counts
    tenant_cnt = coalesce_numeric(df, "# of Tenant Parking")
    public_cnt = coalesce_numeric(df, "# of Public Parking")
    total_spots = tenant_cnt + public_cnt
    # Parking areas (sizes)
    tenant_area = coalesce_numeric(df, "Tenant Area (Sq.m)")
    public_area = coalesce_numeric(df, "Public Area (Sq.m)")
    # Price (if provided, not used for size here)
    public_price = df.get("Public parking price", pd.Series([np.nan]*len(df)))
    # Land use, access, markings
    land_use = df.get("Land Use Purpose", pd.Series([""]*len(df)))
    access_prop = df.get("Access into property", pd.Series([""]*len(df)))
    markings = df.get("Parking space markings", pd.Series([""]*len(df)))
    # Build final frame
    out = pd.DataFrame({
        "Name": df.get("Name"),
        "Cadastral Code": df.get("Cadastral Code"),
        "Address": df.get("Address"),
        "Property Owner": df.get("Property Owner"),
        "Land Use Purpose": land_use,
        "Access into property": access_prop,
        "Parking space markings": markings,
        "Tenant Area (Sq.m)": tenant_area,
        "# of Tenant Parking": tenant_cnt,
        "Public Area (Sq.m)": public_area,
        "# of Public Parking": public_cnt,
        "Public parking price": public_price,
        "Total Spots": total_spots,
        "lat": df["lat"],
        "lon": df["lon"],
    })
    # Drop rows without coordinates
    out = out.dropna(subset=["lat","lon"])
    # Compute radius: scale by total spots (meters for pydeck ScatterplotLayer)
    if len(out) > 0:
        min_s, max_s = out["Total Spots"].min(), out["Total Spots"].max()
        # Avoid zero division
        if max_s == min_s:
            out["radius_m"] = 45.0 + out["Total Spots"] * 0
        else:
            # Map [min_s, max_s] -> [30, 60] meters
            out["radius_m"] = 60.0 + (out["Total Spots"] - min_s) * (100.0 - 60.0) / (max_s - min_s)
    else:
        out["radius_m"] = 75.0
    # Type flags for filtering
    out["Has Tenant"] = out["# of Tenant Parking"] > 0
    out["Has Public"] = out["# of Public Parking"] > 0
    return out

# Sidebar: filters only
with st.sidebar:
    st.header("Filters")
    parking_filter = st.selectbox(
        "Parking Type",
        options=["Both Tenant & Public", "Tenant Only", "Public Only", "All"],
        index=3
    )
    min_spots = int(st.number_input("Minimum total spots", value=0, min_value=0, step=1))
    search_text = st.text_input("Search (by name/address/owner)")

# Load data from bundled CSV
df_raw = load_data("Tbilisi-Parking-Template.csv")

df = prepare_data(df_raw)

# Apply filters
if parking_filter == "Both Tenant & Public":
    mask = df["Has Tenant"] & df["Has Public"]
elif parking_filter == "Tenant Only":
    mask = df["Has Tenant"] & ~df["Has Public"]
elif parking_filter == "Public Only":
    mask = ~df["Has Tenant"] & df["Has Public"]
else:  # "All"
    mask = df["Has Tenant"] | df["Has Public"]

mask &= df["Total Spots"] >= min_spots
if search_text:
    q = search_text.lower()
    mask &= (
        df["Name"].astype(str).str.lower().str.contains(q) |
        df["Address"].astype(str).str.lower().str.contains(q) |
        df["Property Owner"].astype(str).str.lower().str.contains(q)
    )
df_v = df[mask].copy()

# Map view state centered on data
if len(df_v) > 0:
    mid_lat = float(df_v["lat"].mean())
    mid_lon = float(df_v["lon"].mean())
else:
    # Tbilisi center fallback
    mid_lat, mid_lon = 41.7151, 44.8271

INITIAL_VIEW_STATE = pdk.ViewState(
    latitude=mid_lat, longitude=mid_lon, zoom=12, pitch=0
)

# Color by type: both -> strong, tenant only -> one shade, public only -> another
def color_row(row):
    has_t, has_p = row["Has Tenant"], row["Has Public"]
    if has_t and has_p:
        return [0, 255, 0]  # green for both
    elif has_t:
        return [0, 128, 255]  # tenant=blue-ish
    else:
        return [255, 128, 0]  # public=orange-ish

df_v["color"] = df_v.apply(color_row, axis=1)

# Build pydeck layer
layer = pdk.Layer(
    "ScatterplotLayer",
    data=df_v,
    get_position="[lon, lat]",
    get_radius="radius_m",
    get_fill_color="color",
    pickable=True,
    opacity=0.7,
    stroked=True,
    get_line_color=[255, 255, 255],
    line_width_min_pixels=1,
)

# Rich HTML tooltip
tooltip = {
    "html": """
    <div style='font-family: ui-sans-serif, system-ui; font-size: 13px'>
      <div style='font-weight:700; margin-bottom: 4px;'>{Name}</div>
      <div><b>Cadastral:</b> {Cadastral Code}</div>
      <div><b>Address:</b> {Address}</div>
      <div><b>Owner:</b> {Property Owner}</div>
      <hr style='border:none;border-top:1px solid #eee;margin:6px 0' />
      <div><b>Total Spots:</b> {Total Spots}</div>
      <div style='margin-top:4px'><b>Tenant</b>: {# of Tenant Parking} spots, {Tenant Area (Sq.m)} m²</div>
      <div><b>Public</b>: {# of Public Parking} spots, {Public Area (Sq.m)} m²</div>
      <div><b>Public Price</b>: {Public parking price}</div>
      <div style='margin-top:4px'><b>Land Use</b>: {Land Use Purpose}</div>
      <div><b>Access</b>: {Access into property}</div>
      <div><b>Markings</b>: {Parking space markings}</div>
    </div>
    """,
    "style": {"backgroundColor": "white", "color": "black"}
}

r = pdk.Deck(
    initial_view_state=INITIAL_VIEW_STATE,
    layers=[layer],
    tooltip=tooltip,
    map_style="mapbox://styles/mapbox/light-v10",
)

st.pydeck_chart(r, use_container_width=True)

# Add legend
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Filtered Data")
    st.dataframe(df_v.drop(columns=["color"]), use_container_width=True)

with col2:
    st.markdown("### Legend")
    st.markdown("""
    <div style='font-family: ui-sans-serif, system-ui; font-size: 14px'>
      <div style='margin-bottom: 8px'>
        <span style='display: inline-block; width: 20px; height: 20px; background-color: rgb(0, 255, 0); border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>
        <span>Both Tenant & Public</span>
      </div>
      <div style='margin-bottom: 8px'>
        <span style='display: inline-block; width: 20px; height: 20px; background-color: rgb(0, 128, 255); border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>
        <span>Tenant Only</span>
      </div>
      <div style='margin-bottom: 8px'>
        <span style='display: inline-block; width: 20px; height: 20px; background-color: rgb(255, 128, 0); border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>
        <span>Public Only</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
