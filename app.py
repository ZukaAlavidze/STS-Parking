
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from pyproj import Transformer

st.set_page_config(page_title="Tbilisi Parking Map", layout="wide")

st.title("STS 🅿️ Tbilisi Parking")

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
    # Compute radius: scale by total spots (meters for folium)
    if len(out) > 0:
        min_s, max_s = out["Total Spots"].min(), out["Total Spots"].max()
        # Avoid zero division
        if max_s == min_s:
            out["radius_m"] = 45.0 + out["Total Spots"] * 0
        else:
            # Map [min_s, max_s] -> [30, 60] meters
            out["radius_m"] = 30.0 + (out["Total Spots"] - min_s) * (60.0 - 30.0) / (max_s - min_s)
    else:
        out["radius_m"] = 45.0
    # Type flags for filtering
    out["Has Tenant"] = out["# of Tenant Parking"] > 0
    out["Has Public"] = out["# of Public Parking"] > 0
    return out

# Load data from bundled CSV
df_raw = load_data("Tbilisi-Parking-Template.csv")
df = prepare_data(df_raw)

# Create search options from the data
search_options = []
for idx, row in df.iterrows():
    search_options.extend([
        row['Name'],
        row['Address'], 
        row['Property Owner']
    ])

# Remove duplicates and sort
search_options = sorted(list(set([opt for opt in search_options if pd.notna(opt) and str(opt).strip() != ""])))

# Create land use options from the data
land_use_options = df['Land Use Purpose'].dropna().unique()
land_use_options = sorted([opt for opt in land_use_options if str(opt).strip() != ""])

# Sidebar: filters only
with st.sidebar:
    st.header("Filters")
    parking_filter = st.selectbox(
        "Parking Type",
        options=["Both Tenant & Public", "Tenant Only", "Public Only", "All"],
        index=3
    )
    min_spots = int(st.number_input("Minimum total spots", value=0, min_value=0, step=1))
    max_spots = int(st.number_input("Maximum total spots", value=2500, min_value=0, step=1))
    
    # Land use filter
    land_use_filter = st.selectbox(
        "Land Use Purpose",
        options=["All"] + list(land_use_options),
        index=0
    )
    
    # Autocomplete search
    search_text = st.selectbox(
        "Search (by name/address/owner)",
        options=[""] + search_options,
        index=0,
        format_func=lambda x: "Type to search..." if x == "" else x
    )

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
mask &= df["Total Spots"] <= max_spots

# Apply land use filter
if land_use_filter != "All":
    mask &= df["Land Use Purpose"] == land_use_filter

if search_text:
    q = search_text.lower()
    mask &= (
        df["Name"].astype(str).str.lower().str.contains(q) |
        df["Address"].astype(str).str.lower().str.contains(q) |
        df["Property Owner"].astype(str).str.lower().str.contains(q)
    )
df_v = df[mask].copy()

# Create Folium map centered on Heroes Square, Tbilisi
m = folium.Map(
    location=[41.713465, 44.782525],
    zoom_start=14,
    tiles='OpenStreetMap'
)

# Color mapping function
def get_color(row):
    has_t, has_p = row["Has Tenant"], row["Has Public"]
    if has_t and has_p:
        return 'purple'
    elif has_t:
        return 'blue'
    else:
        return 'red'

# Add markers for each parking location
for idx, row in df_v.iterrows():
    # Create popup content
    popup_html = f"""
    <div style='font-family: ui-sans-serif, system-ui; font-size: 13px; width: 300px'>
      <div style='font-weight:700; margin-bottom: 4px;'>{row['Name']}</div>
      <div><b>Cadastral:</b> {row['Cadastral Code']}</div>
      <div><b>Address:</b> {row['Address']}</div>
      <div><b>Owner:</b> {row['Property Owner']}</div>
      <hr style='border:none;border-top:1px solid #eee;margin:6px 0' />
      <div><b>Total Spots:</b> {row['Total Spots']}</div>
      <div style='margin-top:4px'><b>Tenant</b>: {row['# of Tenant Parking']} spots, {row['Tenant Area (Sq.m)']} m²</div>
      <div><b>Public</b>: {row['# of Public Parking']} spots, {row['Public Area (Sq.m)']} m²</div>
      <div><b>Public Price</b>: {row['Public parking price']}</div>
      <div style='margin-top:4px'><b>Land Use</b>: {row['Land Use Purpose']}</div>
      <div><b>Access</b>: {row['Access into property']}</div>
      <div><b>Markings</b>: {row['Parking space markings']}</div>
    </div>
    """
    
    # Add circle marker
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=row['radius_m'] / 10 + 5,  # Increased radius by 5
        popup=folium.Popup(popup_html, max_width=350),
        color='white',
        weight=1,
        fillColor=get_color(row),
        fillOpacity=0.5  # Added transparency (reduced from 0.7 to 0.5)
    ).add_to(m)

# Display the map
st_folium(m, use_container_width=True)

# Add legend
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### Filtered Data")
    st.dataframe(df_v.drop(columns=["color"] if "color" in df_v.columns else []), use_container_width=True)

with col2:
    st.markdown("### Legend")
    st.markdown("""
    <div style='font-family: ui-sans-serif, system-ui; font-size: 14px'>
      <div style='margin-bottom: 8px'>
        <span style='display: inline-block; width: 20px; height: 20px; background-color: purple; border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>
        <span>Both Tenant & Public</span>
      </div>
      <div style='margin-bottom: 8px'>
        <span style='display: inline-block; width: 20px; height: 20px; background-color: blue; border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>
        <span>Tenant Only</span>
      </div>
      <div style='margin-bottom: 8px'>
        <span style='display: inline-block; width: 20px; height: 20px; background-color: red; border-radius: 50%; margin-right: 8px; vertical-align: middle;'></span>
        <span>Public Only</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
