"""
Mitra10 × DHL Trip Integration Dashboard
-----------------------------------------
This Streamlit application analyzes historical DHL shipment data to evaluate
potential route integration opportunities with partner store locations.

The dashboard explores whether existing DHL delivery routes can support
additional pickups or deliveries for nearby partner locations without
requiring new routes.

The analysis combines spatial proximity evaluation, operational truck
availability analysis, and predictive modeling.

Main capabilities:
1. Network Opportunity Analysis
   - Identifies geographic proximity between DHL shipment stops and partner
     store locations.
   - Measures store coverage and route exposure.

2. Operational Route Availability
   - Estimates how many DHL trucks pass near each partner location per day.
   - Analyzes daily and weekly operational patterns.

3. Predictive Truck Availability
   - Uses a machine learning model to estimate future truck availability
     near partner locations based on historical routing patterns.

Key Concepts:
- Stop = unique (tripNumber, destPostcode)
- Overlap occurs when a DHL stop is within a defined radius of a partner store.
- Truck availability is estimated using unique tripNumber counts per day.

Tech Stack:
- Python
- Streamlit
- Pandas / NumPy
- Plotly
- Scikit-learn

Files expected in same folder:
- dhl_stops.csv (columns: tripNumber, destPostcode, des_lat, des_lon, unloadingDate, tariffType, ...)
- mitra10_loc.csv (columns: postalCode, address, lat, lon)

This dashboard serves as a proof-of-concept tool for evaluating data-driven
route integration opportunities within an existing logistics network.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.neighbors import BallTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

EARTH_RADIUS_KM = 6371.0088

STOPS_CSV = "dhl_stops.csv"
STORE_CSV = "mitra10_loc.csv"

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="Mitra10 × DHL Route Overlap", layout="wide")


# -------------------------
# Cached loaders
# -------------------------
@st.cache_data
def load_stops(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Ensure numeric lat/lon
    df["des_lat"] = pd.to_numeric(df["des_lat"], errors="coerce")
    df["des_lon"] = pd.to_numeric(df["des_lon"], errors="coerce")
    df = df.dropna(subset=["des_lat", "des_lon"]).copy()

    # Parse datetime (CSV reload will make it string)
    df["unloadingDate"] = pd.to_datetime(df["unloadingDate"], errors="coerce")
    df = df.dropna(subset=["unloadingDate"]).copy()

    # Normalize tariffType
    if "tariffType" in df.columns:
        df["tariffType"] = df["tariffType"].astype(str)
    else:
        df["tariffType"] = "Unknown"

    return df


@st.cache_data
def load_store(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ["lat", "lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    if "address" not in df.columns:
        df["address"] = df["postalCode"].astype(str)

    return df


@st.cache_data
def build_balltree(latlon_deg: np.ndarray) -> BallTree:
    latlon_rad = np.deg2rad(latlon_deg)
    return BallTree(latlon_rad, metric="haversine")


# -------------------------
# Load data
# -------------------------
stops = load_stops(STOPS_CSV)
stores = load_store(STORE_CSV)

min_date = stops["unloadingDate"].min().date()
max_date = stops["unloadingDate"].max().date()

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

radius_km = st.sidebar.slider("Radius (km)", min_value=1, max_value=10, value=5, step=1)

d1, d2 = st.sidebar.date_input("Date range", value=(min_date, max_date))
if isinstance(d1, (list, tuple)):
    d1, d2 = d1[0], d1[1]
if d1 > d2:
    d1, d2 = d2, d1

tariffs = sorted(stops["tariffType"].dropna().unique().tolist())
tariff_choice = st.sidebar.multiselect(
    "tariffType (multi-select)",
    options=tariffs,
    default=[]
)

use_all_tariff = (len(tariff_choice) == 0)

time_grain = st.sidebar.radio("Trend granularity", ["Monthly", "Daily"], horizontal=True)

trend_mode = st.sidebar.radio(
    "Trend metric",
    ["Store coverage (# stores with overlap)", "Total overlapping stops"],
    horizontal=False
)
ratio_threshold = st.sidebar.slider(
    "Consistency threshold (coverage ratio)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05
)
# Map density control
show_all_stops = st.sidebar.checkbox("Show filtered DHL stop points on map", value=True)
#max_points = st.sidebar.slider("Max DHL points plotted (display only)", 2000, 50000, 15000, step=1000)

# -------------------------
# Filter stops first
# -------------------------
mask = (stops["unloadingDate"].dt.date >= d1) & (stops["unloadingDate"].dt.date <= d2)

# If user selected any tariffType, filter by it
if len(tariff_choice) > 0:
    mask &= stops["tariffType"].isin(tariff_choice)

stops_f = stops.loc[mask].copy()

if len(stops_f) == 0:
    st.error("No stops found for the selected filters. Try expanding date range or choosing a different tariffType.")
    st.stop()

# Downsample for map display only (NOT for calculations)
stops_for_map = stops_f

# -------------------------
# Build spatial index (BallTree) on filtered stops
# -------------------------
tree = build_balltree(stops_f[["des_lat", "des_lon"]].to_numpy())
store_coords_rad = np.deg2rad(stores[["lat", "lon"]].to_numpy())
radius_rad = radius_km / EARTH_RADIUS_KM

# Query: indices of filtered stops within radius for each store
idxs = tree.query_radius(store_coords_rad, r=radius_rad)
# Nearest DHL stop distance (km) for each store (based on currently filtered stops_f)
nearest_dist_rad, _ = tree.query(store_coords_rad, k=1)   # shape (n_stores, 1)
coverage_nearest_km = (nearest_dist_rad[:, 0] * EARTH_RADIUS_KM).astype(float)

# -------------------------
# Build ML dataset: store × date × trucks
# -------------------------
store_daily_rows = []

for store_i, stop_ix in enumerate(idxs):

    if len(stop_ix) == 0:
        continue

    store_name = stores.iloc[store_i]["address"]

    store_stops = stops_f.iloc[stop_ix].copy()

    # normalize date
    store_stops["date"] = store_stops["unloadingDate"].dt.date

    # count unique trucks
    daily_counts = (
        store_stops
        .groupby("date")["tripNumber"]
        .nunique()
        .reset_index()
        .rename(columns={"tripNumber": "trucks"})
    )

    daily_counts["store"] = store_name

    store_daily_rows.append(daily_counts)

ml_df = pd.concat(store_daily_rows, ignore_index=True)

ml_df = ml_df[["store", "date", "trucks"]]


# -------------------------
# Add missing dates (0 trucks)
# -------------------------

all_dates = pd.date_range(d1, d2)

stores_unique = ml_df["store"].unique()

full_index = pd.MultiIndex.from_product(
    [stores_unique, all_dates],
    names=["store", "date"]
)

ml_df = (
    ml_df
    .set_index(["store", "date"])
    .reindex(full_index)
    .fillna(0)
    .reset_index()
)

ml_df["trucks"] = ml_df["trucks"].astype(int)

# -------------------------
# Feature Engineering
# -------------------------

ml_df["date"] = pd.to_datetime(ml_df["date"])

ml_df["weekday"] = ml_df["date"].dt.weekday
ml_df["month"] = ml_df["date"].dt.month
ml_df["week_of_year"] = ml_df["date"].dt.isocalendar().week.astype(int)
ml_df["day_of_month"] = ml_df["date"].dt.day
ml_df["store_id"] = ml_df["store"].astype("category").cat.codes

# -------------------------
# Prepare ML features
# -------------------------
features = [
    "store_id",
    "weekday",
    "month",
    "week_of_year",
    "day_of_month"
]

target = "trucks"

X = ml_df[features]
y = ml_df[target]

# -------------------------
# Train/Test split
# -------------------------

split_date = ml_df["date"].quantile(0.8)

train_df = ml_df[ml_df["date"] <= split_date]
test_df = ml_df[ml_df["date"] > split_date]

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# -------------------------
# Train model
# -------------------------
@st.cache_resource
def train_model(X_train, y_train):

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


model = train_model(X_train, y_train)

# -------------------------
# Feature Importance
# -------------------------
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

# -------------------------
# Predict test set
# -------------------------
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# -------------------------
# Coverage per store
# -------------------------
coverage = stores.copy()
coverage["overlap_stops"] = [len(ix) for ix in idxs]
coverage["nearest_stop_km"] = coverage_nearest_km

covered_store = int((coverage["overlap_stops"] > 0).sum())
total_store = int(len(coverage))
coverage_pct = (covered_store / total_store * 100) if total_store else 0.0

# Avg stops per month in selected date range (rough, for tooltip)
days = max(1, (pd.Timestamp(d2) - pd.Timestamp(d1)).days + 1)
months_equiv = max(1e-6, days / 30.44)
coverage["avg_stops_per_month"] = coverage["overlap_stops"] / months_equiv

# -------------------------
# Trend (Network-level)
# -------------------------
# Prepare bucket axis
bucket_label = "Month" if time_grain == "Monthly" else "Day"

if time_grain == "Monthly":
    all_buckets = pd.date_range(
        pd.Timestamp(d1).to_period("M").start_time,
        pd.Timestamp(d2).to_period("M").start_time,
        freq="MS",
    )
else:
    all_buckets = pd.date_range(pd.Timestamp(d1), pd.Timestamp(d2), freq="D")

# Compute trend
if trend_mode == "Total overlapping stops":
    # union of all overlapping stop indices across stores
    hit_idx = np.unique(np.concatenate([ix for ix in idxs if len(ix) > 0])) if covered_store else np.array([], dtype=int)
    overlap_stops = stops_f.iloc[hit_idx].copy() if len(hit_idx) else stops_f.iloc[0:0].copy()

    if len(overlap_stops) > 0:
        if time_grain == "Monthly":
            overlap_stops["bucket"] = overlap_stops["unloadingDate"].dt.to_period("M").dt.to_timestamp()
        else:
            overlap_stops["bucket"] = overlap_stops["unloadingDate"].dt.floor("D")

        trend = overlap_stops.groupby("bucket").size().reset_index(name="value")
    else:
        trend = pd.DataFrame({"bucket": [], "value": []})

else:
    # Store coverage: #stores that have >=1 overlapping stop per bucket
    edges = []
    for store_i, stop_ix in enumerate(idxs):
        if len(stop_ix) == 0:
            continue
        edges.append(pd.DataFrame({"store_i": store_i, "stop_i": stop_ix}))

    if len(edges) == 0:
        trend = pd.DataFrame({"bucket": [], "value": []})
    else:
        edges = pd.concat(edges, ignore_index=True)

        # attach unloadingDate from stops_f by stop index
        edges = edges.merge(
            stops_f.reset_index(drop=True)[["unloadingDate"]],
            left_on="stop_i",
            right_index=True,
            how="left",
        )

        if time_grain == "Monthly":
            edges["bucket"] = edges["unloadingDate"].dt.to_period("M").dt.to_timestamp()
        else:
            edges["bucket"] = edges["unloadingDate"].dt.floor("D")

        trend = edges.groupby("bucket")["store_i"].nunique().reset_index(name="value")

# Fill missing buckets
trend_full = (
    pd.DataFrame({"bucket": all_buckets})
    .merge(trend, on="bucket", how="left")
    .fillna({"value": 0})
)
trend_full["coverage_ratio"] = (
    trend_full["value"] / total_store if total_store > 0 else 0
)
# Trend KPIs (for coverage mode, these are very meaningful; for stop mode, still useful)
avg_value_per_bucket = float(trend_full["value"].mean()) if len(trend_full) else 0.0
peak_value = int(trend_full["value"].max()) if len(trend_full) else 0
pct_buckets_with_threshold = (
    float((trend_full["coverage_ratio"] >= ratio_threshold).mean() * 100)
    if len(trend_full) else 0.0
)

# -------------------------
# UI: Title
# -------------------------
st.title("Mitra10 × DHL Trip Integration")
st.caption("Metric: Stops = unique (tripNumber, destPostcode). Use sidebar to adjust radius, date range, and tariff type.")

# st.subheader("ML Dataset Preview")
# st.dataframe(ml_df.head(20))

tab1, tab2, tab3 = st.tabs([
    "Network Opportunity",
    "Operational Route Availability",
    "Predictive Truck Availability"
])

with tab1:
    # -------------------------
    # MAP - Top
    # -------------------------
    center_lat = float(coverage["lat"].mean())
    center_lon = float(coverage["lon"].mean())

    fig = go.Figure()

    # DHL stop points (background)
    if show_all_stops:
        fig.add_trace(go.Scattermapbox(
            lat=stops_for_map["des_lat"],
            lon=stops_for_map["des_lon"],
            mode="markers",
            marker=dict(size=4, opacity=0.15),
            name="DHL Stops (filtered)",
        ))

    # Store overlay (colored by stop exposure)
    fig.add_trace(go.Scattermapbox(
        lat=coverage["lat"],
        lon=coverage["lon"],
        mode="markers",
        marker=dict(
            size=12,
            color=coverage["overlap_stops"],
            colorscale="YlOrRd",  # Yellow → Orange → Red
            showscale=True,
            colorbar=dict(title="Stop Exposure"),
        ),
        text=coverage["address"],
        hovertemplate=(
        "<b>%{text}</b><br>"
        "Overlapping Stops: %{customdata[0]}<br>"
        #"Avg Stops/Month (range): %{customdata[1]:.2f}<br>"
        "Nearest DHL Stop: %{customdata[1]:.2f} km"
        "<extra></extra>"
        ),
        customdata=np.column_stack([
        coverage["overlap_stops"],
        #coverage["avg_stops_per_month"],
        coverage["nearest_stop_km"]
        ]),
        name="Mitra10 Stores",
    ))

    fig.update_layout(
        mapbox=dict(
            style="carto-darkmatter",
            zoom=4.8,
            center=dict(lat=center_lat, lon=center_lon),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------
    # KPI row - Middle (aligned with trend)
    # -------------------------
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Store Coverage % (Selected Range)", f"{coverage_pct:.1f}%")
    k2.metric("Covered Stores (Selected Range)", f"{covered_store}/{total_store}")
    #k3.metric(f"Avg Value / {bucket_label}", f"{avg_value_per_bucket:.1f}")
    k3.metric(
        f"{bucket_label} Consistency (≥ {ratio_threshold:.0%} covered)",
        f"{pct_buckets_with_threshold:.1f}%"
    )
    k4.metric(f"Peak Value / {bucket_label}", f"{peak_value}")

    # st.caption(
    #     f"Consistency: {pct_buckets_with_threshold:.1f}% of {bucket_label.lower()}s "
    #     f"have ≥ {ratio_threshold:.0%} coverage."
    # )

    # -------------------------
    # Bottom: Trend + quick table
    # -------------------------
    c_left, c_right = st.columns([2, 1])

    with c_left:

        title = (
            f"{time_grain} Trend — Total Overlapping Stops"
            if trend_mode == "Total overlapping stops"
            else f"{time_grain} Trend — Store Coverage Ratio"
        )
        st.subheader(title)

        # INIT FIGURE
        trend_fig = go.Figure()

        if trend_mode == "Store coverage (# stores with overlap)":

            trend_fig.add_trace(go.Scatter(
                x=trend_full["bucket"],
                y=trend_full["coverage_ratio"],
                mode="lines+markers",
                name="Coverage Ratio",
                customdata=np.column_stack([trend_full["value"]]),
                hovertemplate=
                    "<b>%{x}</b><br>" +
                    "Coverage: %{y:.2%}<br>" +
                    "Covered Stores: %{customdata[0]}<extra></extra>"
            ))

            trend_fig.update_yaxes(
                title_text="Coverage Ratio",
                tickformat=".0%",
                range=[0, 1]
            )

        else:

            trend_fig.add_trace(go.Scatter(
                x=trend_full["bucket"],
                y=trend_full["value"],
                mode="lines+markers",
                name="Overlapping Stops"
            ))

            trend_fig.update_yaxes(title_text="Overlapping Stops")

        trend_fig.update_xaxes(title_text=bucket_label)
        trend_fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=320)

        st.plotly_chart(trend_fig, use_container_width=True)

    with c_right:
        st.subheader("Top Stores by Stop Exposure")
        top_n = 10 if len(coverage) >= 10 else len(coverage)
        top_tbl = coverage.sort_values("overlap_stops", ascending=False).head(top_n)[
            ["address", "overlap_stops", "avg_stops_per_month"]
        ]
        st.dataframe(top_tbl, use_container_width=True, hide_index=True)

    # -------------------------
    # Business Impact Simulation (Volume Uplift)
    # -------------------------

    st.divider()
    st.header("Business Impact Simulation (Scenario-Based)")

    # Baseline total shipment (unique tripNumber)
    total_shipments = stops_f["tripNumber"].nunique()

    # Total potential overlap stops (network-level union)
    hit_idx = np.unique(np.concatenate([ix for ix in idxs if len(ix) > 0])) if covered_store else np.array([], dtype=int)
    total_overlap_stops = len(hit_idx)

    b1, b2 = st.columns(2)
    b1.metric("Total Shipments (Selected Range)", total_shipments)
    b2.metric("Total Potential Overlap Stops", total_overlap_stops)

    st.markdown("### Volume Frequency Uplift Scenarios")

    scenario_rates = [0.10, 0.20, 0.30]

    scenario_data = []

    for rate in scenario_rates:
        additional_drops = total_overlap_stops * rate
        uplift_pct = (additional_drops / total_shipments * 100) if total_shipments > 0 else 0

        scenario_data.append({
            "Scenario": f"{int(rate*100)}% Conversion",
            "Conversion Rate": f"{int(rate*100)}%",
            "Additional Drops": int(additional_drops),
            "Volume Freq Uplift (%)": f"{uplift_pct:.2f}%"
        })

    scenario_df = pd.DataFrame(scenario_data)

    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

    # bar chart visualization
    st.markdown("### Visualized Volume Frequency Uplift")

    bar_fig = go.Figure()

    bar_fig.add_trace(go.Bar(
        x=scenario_df["Scenario"],
        y=[float(x.replace("%","")) for x in scenario_df["Volume Freq Uplift (%)"]],
        text=scenario_df["Volume Freq Uplift (%)"],
        textposition="outside"
    ))

    bar_fig.update_layout(
        yaxis_title="Volume Freq Uplift (%)",
        xaxis_title="Scenario",
        height=350
    )

    st.plotly_chart(bar_fig, use_container_width=True)

    st.caption(
        "Assumptions: 1 overlapping stop ≈ 1 potential incremental shipment. "
        "Conversion rate reflects operational alignment and partner readiness. "
        "Cost impact not yet incorporated."
    )
with tab2:

    st.header("Operational Route Availability")

    st.caption(
        "This analysis estimates how many DHL trucks pass near each partner location per day, "
        "based on existing shipment routes."
    )

    st.subheader("Operational Date Range")
    
    tab2_date_range = st.date_input(
        "Select Operational Date Range",
        value=(stops_f["unloadingDate"].min().date(), stops_f["unloadingDate"].max().date()),
        key="tab2_date_range"
    )

    if len(tab2_date_range) != 2:
        st.stop()

    tab2_start, tab2_end = pd.to_datetime(tab2_date_range[0]), pd.to_datetime(tab2_date_range[1])

    st.subheader("Unloading Time Window")

    col1, col2 = st.columns(2)

    with col1:
        start_hour = st.selectbox(
            "Start Time",
            options=list(range(0,24)),
            format_func=lambda x: f"{x:02d}:00",
            index=6
        )

    with col2:
        end_hour = st.selectbox(
            "End Time",
            options=list(range(0,24)),
            format_func=lambda x: f"{x:02d}:00",
            index=10
        )

    st.info(
        "Hour-level unloading time is not available in the current dataset. "
        "When timestamp data is provided, the dashboard will filter trucks based on this selected time window."
    )
    # -----------------------------------
    # Build edge table (store - stop)
    # -----------------------------------
    edges = []

    for store_i, stop_ix in enumerate(idxs):
        if len(stop_ix) == 0:
            continue

        tmp = pd.DataFrame({
            "store_i": store_i,
            "stop_i": stop_ix
        })

        edges.append(tmp)

    if len(edges) == 0:
        st.warning("No overlapping routes found for the selected filters.")
        st.stop()

    edges = pd.concat(edges, ignore_index=True)

    edges = edges.merge(
        stops_f.reset_index(drop=True)[["tripNumber","unloadingDate"]],
        left_on="stop_i",
        right_index=True,
        how="left"
    )

    edges = edges[
        (edges["unloadingDate"] >= tab2_start) &
        (edges["unloadingDate"] <= tab2_end)
    ]

    edges["store"] = edges["store_i"].map(stores["address"])
    edges["date"] = edges["unloadingDate"].dt.floor("D")
    edges["weekday"] = edges["unloadingDate"].dt.day_name()

    # -----------------------------------
    # Trucks per store per day
    # -----------------------------------
    trucks_daily = (
        edges.groupby(["store","date"])
        .agg(trucks=("tripNumber","nunique"))
        .reset_index()
    )

    # -----------------------------------
    # Average trucks per weekday
    # -----------------------------------
    trucks_weekday = (
        trucks_daily.assign(
            weekday=pd.to_datetime(trucks_daily["date"]).dt.day_name()
        )
        .groupby(["store","weekday"])
        .agg(avg_trucks=("trucks","mean"))
        .reset_index()
    )

    st.subheader("Average Trucks Passing Nearby per Store per Day")

    pivot = trucks_weekday.pivot(
        index="store",
        columns="weekday",
        values="avg_trucks"
    )

    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

    pivot = pivot.reindex(columns=[c for c in weekday_order if c in pivot.columns])

    fig_heatmap = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )

    fig_heatmap.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Store",
        height=450
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # -----------------------------------
    # Store-level analysis
    # -----------------------------------

    st.subheader("Store-Level Route Availability")

    store_select = st.selectbox(
        "Select store",
        sorted(trucks_daily["store"].unique())
    )
    st.caption(
        f"Analysis period: {tab2_start.date()} → {tab2_end.date()}"
    )
    store_df = trucks_daily[trucks_daily["store"] == store_select].copy()
    store_df["weekday"] = pd.to_datetime(store_df["date"]).dt.day_name()
    store_df["week"] = pd.to_datetime(store_df["date"]).dt.isocalendar().week
    # buat full date range
    all_dates = pd.date_range(tab2_start, tab2_end, freq="D")
    store_df = (
        pd.DataFrame({"date": all_dates})
        .merge(store_df, on="date", how="left")
        .fillna({"trucks": 0})
    )

    store_df["store"] = store_select

    fig_store = px.line(
        store_df,
        x="date",
        y="trucks",
        markers=True,
        title=f"Daily Trucks Passing Near {store_select}"
    )
    fig_store.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="Number of Trucks",
    )

    fig_store.update_traces(
        line=dict(width=2)
    )

    st.plotly_chart(fig_store, use_container_width=True)

    st.metric(
        "Average Trucks per Day",
        f"{store_df['trucks'].mean():.2f}"
    )

    st.subheader("Weekly Truck Availability")

    mode = st.radio(
        "Choose aggregation mode:",
        ["Monthly Total per Weekday", "Weekly Breakdown"],
        horizontal=True
    )

    store_df["weekday"] = pd.to_datetime(store_df["date"]).dt.day_name()

    weekday_order = [
        "Monday","Tuesday","Wednesday",
        "Thursday","Friday","Saturday","Sunday"
    ]

    # -------------------------
    # MODE 1 — Total per Day
    # -------------------------
    if mode == "Monthly Total per Weekday":
        weekday_pattern = (
            store_df
            .groupby("weekday")
            .agg(total_trucks=("trucks","sum"))
            .reset_index()
        )

        weekday_pattern["weekday"] = pd.Categorical(
            weekday_pattern["weekday"],
            categories=weekday_order,
            ordered=True
        )

        weekday_pattern = weekday_pattern.sort_values("weekday")

        fig_weekday = px.bar(
            weekday_pattern,
            x="weekday",
            y="total_trucks",
            text_auto=True,
            title=f"Total Trucks by Day of Week — {store_select}"
        )
        
    # -------------------------
    # MODE 2 — Weekly Breakdown
    # -------------------------
    else:

        store_df["week"] = pd.to_datetime(store_df["date"]).dt.isocalendar().week

        week_ranges = (
            store_df
            .groupby("week")
            .agg(
                start_date=("date","min"),
                end_date=("date","max")
            )
            .reset_index()
        )

        week_ranges["label"] = week_ranges.apply(
            lambda x: f"Week {x['week']} ({x['start_date'].strftime('%b %d')} – {x['end_date'].strftime('%b %d')})",
            axis=1
        )

        week_options = dict(zip(week_ranges["label"], week_ranges["week"]))

        selected_label = st.selectbox(
            "Select week",
            list(week_options.keys())
        )

        selected_week = week_options[selected_label]

        week_df = store_df[store_df["week"] == selected_week].copy()

        week_pattern = (
            week_df
            .groupby("weekday")
            .agg(trucks=("trucks","sum"))
            .reset_index()
        )

        week_pattern["weekday"] = pd.Categorical(
            week_pattern["weekday"],
            categories=weekday_order,
            ordered=True
        )

        week_pattern = week_pattern.sort_values("weekday")

        fig_weekday = px.bar(
            week_pattern,
            x="weekday",
            y="trucks",
            text_auto=True,
            title=selected_label
        )

    # -------------------------
    # Plot chart (ALL MODES)
    # -------------------------
    fig_weekday.update_layout(
        xaxis_title="Day of Week",
        yaxis_title="Trucks",
        height=350
    )

    st.plotly_chart(fig_weekday, use_container_width=True)

with tab3:

    st.header("Predictive Truck Availability")

    st.caption(
        "Predictive model estimating truck availability "
        "based on historical routing patterns."
    )

    st.subheader("Predict Truck Availability")

    store_list = ml_df["store"].unique()

    col1, col2 = st.columns(2)

    selected_store = col1.selectbox(
        "Select Store",
        store_list
    )

    date_range = col2.date_input(
        "Select Date Range",
        value=(ml_df["date"].min(), ml_df["date"].max())
    )

    if len(date_range) == 2:

        start_date = pd.to_datetime(date_range[0])
        end_date = pd.to_datetime(date_range[1])

        dates = pd.date_range(start_date, end_date)

        store_id = (
            ml_df[ml_df["store"] == selected_store]["store_id"]
            .iloc[0]
        )

        predictions = []

        for d in dates:

            input_df = pd.DataFrame({
                "store_id":[store_id],
                "weekday":[d.weekday()],
                "month":[d.month],
                "week_of_year":[d.isocalendar().week],
                "day_of_month":[d.day]
            })

            pred = model.predict(input_df)[0]

            predictions.append({
                "date": d,
                "weekday": d.day_name(),
                "predicted_trucks": pred
            })

        pred_df = pd.DataFrame(predictions)

        # -------------------------
        # Line Chart (Predicted per Date)
        # -------------------------

        st.subheader("Predicted Trucks per Day")

        fig_line = px.line(
            pred_df,
            x="date",
            y="predicted_trucks",
            markers=True,
            title=f"Predicted Trucks — {selected_store}"
        )

        fig_line.update_layout(
            xaxis_title="Date",
            yaxis_title="Predicted Trucks"
        )

        st.plotly_chart(
            fig_line,
            use_container_width=True,
            key="prediction_line_chart"
        )

        # -------------------------
        # Weekday Aggregation
        # -------------------------

        st.subheader("Predicted Weekly Pattern")

        mode = st.radio(
            "Aggregation Mode",
            ["Monthly Total per Weekday", "Weekly Breakdown"],
            horizontal=True
        )

        pred_df["weekday"] = pd.to_datetime(pred_df["date"]).dt.day_name()

        weekday_order = [
            "Monday","Tuesday","Wednesday",
            "Thursday","Friday","Saturday","Sunday"
        ]

        if mode == "Monthly Total per Weekday":

            weekday_pattern = (
                pred_df
                .groupby("weekday")
                .agg(trucks=("predicted_trucks","sum"))
                .reset_index()
            )

        else:

            pred_df["week"] = pd.to_datetime(pred_df["date"]).dt.isocalendar().week

            week_ranges = (
                pred_df
                .groupby("week")
                .agg(
                    start_date=("date","min"),
                    end_date=("date","max")
                )
                .reset_index()
            )

            week_ranges["label"] = week_ranges.apply(
                lambda x: f"Week {x['week']} ({x['start_date'].strftime('%b %d')} – {x['end_date'].strftime('%b %d')})",
                axis=1
            )

            week_options = dict(zip(week_ranges["label"], week_ranges["week"]))

            selected_label = st.selectbox(
                "Select Week",
                list(week_options.keys())
            )

            selected_week = week_options[selected_label]

            week_df = pred_df[pred_df["week"] == selected_week].copy()

            weekday_pattern = (
                week_df
                .groupby("weekday")
                .agg(trucks=("predicted_trucks","sum"))
                .reset_index()
            )

        weekday_pattern["weekday"] = pd.Categorical(
            weekday_pattern["weekday"],
            categories=weekday_order,
            ordered=True
        )

        weekday_pattern = weekday_pattern.sort_values("weekday")

        fig_bar = px.bar(
            weekday_pattern,
            x="weekday",
            y="trucks",
            text_auto=".2f",
            title="Predicted Trucks by Weekday"
        )

        fig_bar.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Predicted Trucks"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True,
            key="prediction_weekday_chart"
        )


    st.divider()
    st.subheader("Model Performance")

    c1, c2 = st.columns(2)

    c1.metric("MAE", f"{mae:.2f} trucks")
    c2.metric("RMSE", f"{rmse:.2f} trucks")
    st.subheader("Actual vs Predicted Trucks")

    # store selector khusus untuk performance chart
    perf_store_options = ["All Stores"] + sorted(test_df["store"].unique())

    selected_perf_store = st.selectbox(
        "Select Store for Performance View",
        perf_store_options
    )

    test_plot = test_df.copy()
    test_plot["predicted"] = y_pred

    if selected_perf_store != "All Stores":
        test_plot = test_plot[test_plot["store"] == selected_perf_store]

    fig = px.line(
        test_plot,
        x="date",
        y=["trucks", "predicted"],
        title=f"Actual vs Predicted Trucks — {selected_perf_store}"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Trucks"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="ml_actual_vs_predicted_chart"
    )

    # st.subheader("Feature Importance")

    fig_importance = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        text_auto=".2f",
        title="Model Feature Importance"
    )

    fig_importance.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature"
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True,
        key="feature_importance_chart"
    )
# -------------------------
# Download
# -------------------------
st.divider()
# st.download_button(
#     label="Download store overlap table (CSV)",
#     data=coverage.to_csv(index=False).encode("utf-8"),
#     file_name="mitra10_store_overlap.csv",
#     mime="text/csv",
# )