# Mitra10 × DHL Trip Integration Analysis (Case Study) Dashboard

An interactive data analytics and predictive modeling dashboard to evaluate **partner route integration opportunities within DHL’s existing logistics network**.

The dashboard analyzes historical shipment data to determine whether partner distribution points can leverage existing DHL routes for additional pickups or deliveries without requiring new routes.

Built with **Python, Streamlit, Plotly, and Scikit-learn**.

---

# Project Overview

In logistics operations, external partners may seek to integrate their shipments into existing delivery routes when their locations are geographically close to the network.

However, before such integration can be operationalized, several key questions must be addressed:

- Are partner locations geographically close to existing DHL shipment routes?
- How frequently do DHL trucks pass near those locations?
- Are these operational patterns stable over time?
- Can future truck availability near partner locations be estimated?

This project answers those questions through a **three-layer analytical approach**:

1. **Network-level spatial analysis**
2. **Operational truck availability analysis**
3. **Predictive truck availability modeling**

The result is an interactive **Streamlit dashboard** that allows exploration of route integration feasibility.

---

# Dashboard Preview

The dashboard is structured into **three analytical modules**:

1. Network Opportunity
2. Operational Route Availability
3. Predictive Truck Availability

Each module answers a different operational question regarding route integration.

---

# Data Sources

The analysis uses two primary datasets.

## 1. DHL Shipment Stop Data

This dataset contains historical shipment stop information.

Key fields include:

| Column | Description |
|------|-------------|
| tripNumber | Unique truck trip identifier |
| destPostcode | Destination postal code |
| des_lat / des_lon | Destination coordinates |
| unloadingDate | Shipment unloading date |
| tariffType | Shipment type |
| transporter | Logistics provider |

The main operational metric used in the analysis is:
Stop = unique(tripNumber, destPostcode)

This represents a unique unloading stop within a trip.

---

## 2. Partner Store Location Data

Contains the geographical locations of partner stores.

Key fields include:

| Column | Description |
|------|-------------|
| address | Store location name |
| postalCode | Postal code |
| lat / lon | Store coordinates |

These coordinates are used to measure **spatial proximity to DHL shipment stops**.

---

# Analytical Methodology

The project follows a structured analytical workflow.

## 1. Spatial Proximity Matching

Partner store locations are matched with DHL shipment stops using **radius-based spatial matching**.

For each store: Store is considered "covered" if a DHL stop exists within the defined radius.

Users can dynamically adjust the proximity radius within the dashboard.

---

## 2. Stop Exposure Measurement

If a DHL stop falls within the radius of a store, it is counted as an **overlapping stop event**.

From these events, the dashboard calculates:

- number of overlapping stops per store
- monthly exposure trends
- spatial coverage ratios

This helps identify which locations are **most frequently exposed to DHL routes**.

---

## 3. Operational Truck Availability Analysis

The next step is estimating **actual truck availability near partner locations**.

Operational truck availability is calculated as:
Unique tripNumber per store per date


This represents the number of trucks passing near a store on a given day.

This analysis allows users to evaluate:

- daily truck availability
- weekly operational patterns
- variability across different store locations

---

## 4. Scenario-Based Business Impact Simulation

To estimate the business impact of route integration, a simple scenario-based model is applied.

Potential Additional Shipments =
Overlapping Stops × Conversion Rate


Where conversion rate represents the percentage of overlaps that could realistically become integrated shipments.

Example scenarios:

| Conversion Rate | Estimated Shipment Uplift |
|----------------|---------------------------|
| 10% | Conservative adoption |
| 20% | Moderate integration |
| 30% | High integration |

This simulation provides a rough estimate of **potential shipment volume growth**.

---

## 5. Predictive Truck Availability (Machine Learning)

An experimental machine learning model is implemented to estimate **future truck availability near partner locations**.

The modeling dataset aggregates historical data into:
store × date → number of trucks


Feature engineering includes:

- store_id
- weekday
- week_of_year
- day_of_month
- month

The model learns operational patterns to estimate expected truck availability for future dates.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# Dashboard Architecture

The dashboard is built using the following technologies:

| Tool | Purpose |
|-----|--------|
| Python | Core programming language |
| Streamlit | Interactive dashboard framework |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Plotly | Interactive visualizations |
| Scikit-learn | Machine learning modeling |

---

# Dashboard Structure

## Tab 1 — Network Opportunity

Evaluates **geographic compatibility between DHL routes and partner store locations**.

Main components:

- Spatial network map
- Store coverage metrics
- Monthly coverage trends
- Top stores by route exposure
- Scenario-based shipment uplift simulation

Key question answered:

> Are DHL routes already geographically close to partner locations?

---

## Tab 2 — Operational Route Availability

Analyzes **actual truck traffic near partner locations**.

Main components:

- Heatmap of average trucks per store per weekday
- Store-level daily truck analysis
- Weekly truck availability patterns

Key question answered:

> How many trucks pass near each partner location on specific days?

---

## Tab 3 — Predictive Truck Availability

Implements a **machine learning model** to estimate future truck availability.

Main components:

- Predicted trucks per day
- Predicted weekly pattern
- Model performance metrics
- Feature importance analysis

Key question answered:

> Can truck availability near partner locations be predicted using historical patterns?

---

# Key Insights

From the analysis, several insights can be derived:

- A large proportion of partner store locations fall within proximity of existing DHL routes.
- Route overlap occurs consistently across operational months.
- Certain locations have significantly higher operational exposure.
- Integrating shipments into existing routes may increase shipment volume without expanding the network.
- Operational patterns are structured enough to allow predictive modeling.

---

# Potential Future Improvements

Several enhancements could further improve the analysis:

- Incorporating **truck capacity constraints**
- Adding **unloading time window data**
- Including **shipment weight or volume classes**
- Integrating **route scheduling data**
- Applying more advanced time-series forecasting models

---

# Author

Mayra


AI & Automation Support Intern - DHL Supply Chain
