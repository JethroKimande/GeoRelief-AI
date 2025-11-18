"""
Streamlit Dashboard for GeoRelief-AI
Interactive visualization and analytics platform for humanitarian resource allocation
"""

import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium  # type: ignore
import plotly.express as px
from pathlib import Path
import sys
from typing import Optional, Union, Any

# Add parent directory to path to import core module
sys.path.insert(0, str(Path(__file__).resolve().parent))

from core import config
from core.model_engine import get_model_engine

# Page configuration
st.set_page_config(
    page_title="GeoRelief-AI | Humanitarian Intelligence Platform",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_engine' not in st.session_state:
    st.session_state.model_engine = None
if 'master_gdf' not in st.session_state:
    st.session_state.master_gdf = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

@st.cache_resource
def load_model():
    """Load the model engine (cached)"""
    model_engine = get_model_engine()
    if model_engine.load_model():
        return model_engine
    
    # If model doesn't exist, return None (will show setup instructions)
    return None

@st.cache_data
def load_data():
    """Load the master dataset (cached)"""
    if config.MASTER_DATASET_PATH.exists():
        try:
            return gpd.read_file(config.MASTER_DATASET_PATH)
        except Exception as e:
            st.error(f"Error loading data file: {e}")
            return None
    return None

def get_color(score: float, min_score: float, max_score: float) -> str:
    """Get color based on priority score"""
    if max_score == min_score:
        return '#91cf60'
    
    normalized: float = (score - min_score) / (max_score - min_score) * 100
    
    if normalized > 80:
        return '#d73027'  # Critical - Red
    elif normalized > 60:
        return '#fc8d59'  # High - Orange
    elif normalized > 40:
        return '#fee08b'  # Medium - Yellow
    elif normalized > 20:
        return '#d9ef8b'  # Low - Light Green
    else:
        return '#91cf60'  # Very Low - Green

def create_map(gdf: gpd.GeoDataFrame, predictions: pd.Series) -> folium.Map:  # type: ignore
    """Create an interactive Folium map"""
    # Calculate center
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='CartoDB positron'
    )
    
    # Add satellite layer option
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add CartoDB layer
    folium.TileLayer(
        tiles='CartoDB positron',
        name='Analytic Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Calculate min/max for color scaling
    min_score = predictions.min()
    max_score = predictions.max()
    
    # Add GeoJSON with styling
    for idx, row in gdf.iterrows():
        score = predictions.iloc[idx] if isinstance(predictions, pd.Series) else predictions[idx]
        color = get_color(score, min_score, max_score)
        
        # Create popup content
        popup_html = f"""
        <div style="width: 250px;">
            <h4 style="margin: 0 0 10px 0; color: #2c3e50;">{row.get('ADM1_EN', row.get('NAME_1', 'Unknown Region'))}</h4>
            <table style="width: 100%; font-size: 12px;">
                <tr>
                    <td><b>Priority Score:</b></td>
                    <td style="color: {color}; font-weight: bold;">{score:.2f}</td>
                </tr>
                <tr>
                    <td><b>Pop. Density:</b></td>
                    <td>{int(row.get('population_density', 0)):,} /km²</td>
                </tr>
                <tr>
                    <td><b>Vulnerability:</b></td>
                    <td>{row.get('vulnerability_index', 0):.2f}</td>
                </tr>
                <tr>
                    <td><b>Flood Risk:</b></td>
                    <td>{(row.get('base_flood_risk', 0) * 100):.1f}%</td>
                </tr>
        """
        
        if 'river_risk_score' in row and pd.notna(row['river_risk_score']):
            popup_html += f"""
                <tr>
                    <td><b>River Risk:</b></td>
                    <td>{row['river_risk_score']:.2f}</td>
                </tr>
            """
        
        if 'soil_saturation' in row and pd.notna(row['soil_saturation']):
            popup_html += f"""
                <tr>
                    <td><b>Soil Saturation:</b></td>
                    <td>{row['soil_saturation']:.2f}</td>
                </tr>
            """
        
        popup_html += "</table></div>"
        
        # Add feature to map
        folium.GeoJson(
            row.geometry,
            style_function=lambda feature, score=score, color=color: {
                'fillColor': color,
                'color': 'white',
                'weight': 1,
                'fillOpacity': 0.7,
                'dashArray': '3'
            },
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row.get('ADM1_EN', row.get('NAME_1', 'Unknown'))}: {score:.2f}"
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <h4 style="margin: 0 0 10px 0;">Priority Score</h4>
    <p style="margin: 2px 0;"><i style="background:#d73027; width: 16px; height: 16px; display: inline-block; margin-right: 5px;"></i> Critical (80-100)</p>
    <p style="margin: 2px 0;"><i style="background:#fc8d59; width: 16px; height: 16px; display: inline-block; margin-right: 5px;"></i> High (60-80)</p>
    <p style="margin: 2px 0;"><i style="background:#fee08b; width: 16px; height: 16px; display: inline-block; margin-right: 5px;"></i> Medium (40-60)</p>
    <p style="margin: 2px 0;"><i style="background:#d9ef8b; width: 16px; height: 16px; display: inline-block; margin-right: 5px;"></i> Low (20-40)</p>
    <p style="margin: 2px 0;"><i style="background:#91cf60; width: 16px; height: 16px; display: inline-block; margin-right: 5px;"></i> Very Low (0-20)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main() -> None:
    # Header
    st.markdown('<h1 class="main-header">🌍 GeoRelief-AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Humanitarian Intelligence Platform | Real-time Disaster Prioritization Engine</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://sdgs.un.org/sites/default/files/goals/E_SDG_Icons-11.jpg", width=50)
        st.markdown("### Navigation")
        page = st.radio(
            "Select View",
            ["📊 Dashboard", "🗺️ Interactive Map", "📈 Analytics", "ℹ️ About"]
        )
        
        st.markdown("---")
        st.markdown("### 📤 Upload Real Data Files")
        st.markdown("**Upload your trained model and processed data:**")
        
        uploaded_model = st.file_uploader("Upload Model (.h5)", type=['h5'], key='model_upload', help="Upload your trained TensorFlow model file")
        uploaded_scaler = st.file_uploader("Upload Scaler (.pkl)", type=['pkl'], key='scaler_upload', help="Upload your StandardScaler pickle file")
        uploaded_data = st.file_uploader("Upload Dataset (.geojson)", type=['geojson'], key='data_upload', help="Upload your processed GeoJSON dataset")
        
        if uploaded_model or uploaded_scaler or uploaded_data:
            if st.button("💾 Save Uploaded Files", type="primary"):
                # Create directories if they don't exist
                config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
                config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
                
                saved_files = []
                if uploaded_model:
                    model_path = config.MODEL_PATH
                    with open(model_path, "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    saved_files.append(f"✅ Model saved: {str(model_path)}")
                    st.session_state.model_engine = None  # Reset to reload
                    st.cache_resource.clear()  # Clear cache
                
                if uploaded_scaler:
                    scaler_path = config.SCALER_PATH
                    with open(scaler_path, "wb") as f:
                        f.write(uploaded_scaler.getbuffer())
                    saved_files.append(f"✅ Scaler saved: {str(scaler_path)}")
                    st.session_state.model_engine = None  # Reset to reload
                    st.cache_resource.clear()  # Clear cache
                
                if uploaded_data:
                    data_path = config.MASTER_DATASET_PATH
                    with open(data_path, "wb") as f:
                        f.write(uploaded_data.getbuffer())
                    saved_files.append(f"✅ Data saved: {str(data_path)}")
                    st.session_state.master_gdf = None  # Reset to reload
                    st.cache_data.clear()  # Clear cache
                
                for msg in saved_files:
                    st.success(msg)
                st.info("🔄 Reloading app with your data...")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### SDG Alignment")
        st.markdown("""
        - **SDG 11:** Sustainable Cities
        - **SDG 13:** Climate Action
        - **SDG 6:** Clean Water
        - **SDG 3:** Good Health
        """)
    
    # Load data
    with st.spinner("Loading AI models and data..."):
        if st.session_state.model_engine is None:
            st.session_state.model_engine = load_model()
        
        if st.session_state.master_gdf is None:
            st.session_state.master_gdf = load_data()
    
    # Check if data is loaded
    if st.session_state.model_engine is None or st.session_state.master_gdf is None:
        st.error("⚠️ Model or data not found")
        
        st.markdown("### 📋 How to Use Real Data")
        st.markdown("""
        #### 🚀 Quick Start: Upload Files (Easiest)
        
        **Use the file upload section in the sidebar** (left side) to upload:
        1. Your trained model file: `models/priority_model.h5`
        2. Your scaler file: `models/scaler.pkl`
        3. Your processed dataset: `processed_data/global_master_dataset.geojson`
        
        After uploading, click **"💾 Save Uploaded Files"** and the app will reload with your data.
        
        #### 📥 Generate Your Own Data (Local Setup)
        
        If you haven't created the model and data files yet, follow these steps locally:
        
        **Step 1: Download Data**
        ```bash
        python scripts/download_data.py
        python scripts/download_hydrology.py
        ```
        
        **Step 2: Process Data**
        ```bash
        python -m core.data_processor
        ```
        This creates: `processed_data/global_master_dataset.geojson`
        
        **Step 3: Train Model**
        ```bash
        python scripts/1_train_model.py
        ```
        This creates: `models/priority_model.h5` and `models/scaler.pkl`
        
        **Step 4: Upload to Streamlit Cloud**
        - Use the file upload section in the sidebar
        - Or commit files to your repository (if not too large)
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Documentation")
        st.markdown("""
        - **README.md**: Complete setup guide
        - **DATA_ACQUISITION_GUIDE.md**: Detailed data download instructions
        - **QUICK_START_REAL_DATA.md**: Quick start guide
        """)
        
        st.info("💡 **Tip**: For local development, ensure you've completed the setup steps above. For Streamlit Cloud, you'll need to upload the model and data files or configure cloud storage access.")
        
        return
    
    gdf = st.session_state.master_gdf.copy()
    
    # Generate predictions
    if st.session_state.predictions is None:
        X = gdf[config.STANDARD_FEATURES].fillna(0)
        predictions = st.session_state.model_engine.predict(X)
        st.session_state.predictions = pd.Series(predictions, index=gdf.index)
    
    predictions = st.session_state.predictions
    gdf['Predicted_Priority_Score'] = predictions
    
    # Normalize scores for display (0-100)
    min_score = predictions.min()
    max_score = predictions.max()
    if max_score > min_score:
        gdf['Priority_Score_Normalized'] = ((predictions - min_score) / (max_score - min_score) * 100)
    else:
        gdf['Priority_Score_Normalized'] = 50.0
    
    # Page routing
    if page == "📊 Dashboard":
        show_dashboard(gdf, predictions)
    elif page == "🗺️ Interactive Map":
        show_map(gdf, predictions)
    elif page == "📈 Analytics":
        show_analytics(gdf, predictions)
    elif page == "ℹ️ About":
        show_about()

def show_dashboard(gdf: gpd.GeoDataFrame, predictions: pd.Series) -> None:  # type: ignore
    """Main dashboard view"""
    st.markdown("## 📊 Mission Control Dashboard")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Regions Analyzed",
            value=len(gdf),
            delta=None
        )
    
    with col2:
        high_risk = (gdf['Priority_Score_Normalized'] > 60).sum()
        st.metric(
            label="Regions at Critical Risk",
            value=int(high_risk),
            delta=f"{(high_risk/len(gdf)*100):.1f}%"
        )
    
    with col3:
        avg_vuln = gdf['vulnerability_index'].mean()
        st.metric(
            label="Average Vulnerability Index",
            value=f"{avg_vuln:.2f}",
            delta=None
        )
    
    with col4:
        total_pop = gdf['population_density'].sum()
        st.metric(
            label="Total Population Density",
            value=f"{int(total_pop):,}",
            delta=None
        )
    
    st.markdown("---")
    
    # Top Priority Regions
    st.markdown("### 🚨 Top Priority Regions")
    top_regions = gdf.nlargest(10, 'Predicted_Priority_Score')[
        ['ADM1_EN', 'NAME_1', 'Predicted_Priority_Score', 'Priority_Score_Normalized', 
         'population_density', 'vulnerability_index', 'base_flood_risk']
    ].copy()
    
    # Rename columns for display
    top_regions.columns = ['Admin Name', 'Name', 'Priority Score', 'Normalized Score', 
                          'Pop. Density', 'Vulnerability', 'Flood Risk']
    top_regions['Normalized Score'] = top_regions['Normalized Score'].round(1)
    top_regions['Pop. Density'] = top_regions['Pop. Density'].round(0).astype(int)
    top_regions['Vulnerability'] = top_regions['Vulnerability'].round(2)
    top_regions['Flood Risk'] = (top_regions['Flood Risk'] * 100).round(1)
    
    st.dataframe(
        top_regions,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Priority Score Distribution")
        fig = px.histogram(
            gdf,
            x='Priority_Score_Normalized',
            nbins=30,
            labels={'Priority_Score_Normalized': 'Priority Score (Normalized)', 'count': 'Number of Regions'},
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Vulnerability vs Priority Score")
        fig = px.scatter(
            gdf,
            x='vulnerability_index',
            y='Priority_Score_Normalized',
            hover_data=['ADM1_EN', 'NAME_1'],
            labels={'vulnerability_index': 'Vulnerability Index', 
                   'Priority_Score_Normalized': 'Priority Score (Normalized)'},
            color='Priority_Score_Normalized',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig, use_container_width=True)

def show_map(gdf: gpd.GeoDataFrame, predictions: pd.Series) -> None:  # type: ignore
    """Interactive map view"""
    st.markdown("## 🗺️ Interactive Priority Map")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        min_score_filter = st.slider(
            "Minimum Priority Score",
            min_value=float(predictions.min()),
            max_value=float(predictions.max()),
            value=float(predictions.min()),
            step=0.1
        )
    
    with col2:
        # Satellite layer toggle is handled in create_map function
        _ = st.checkbox("Show Satellite Layer", value=False, disabled=True, help="Satellite layer can be toggled on the map itself")
    
    # Filter data
    filtered_gdf = gdf[gdf['Predicted_Priority_Score'] >= min_score_filter].copy()
    filtered_predictions = predictions[filtered_gdf.index]
    
    if len(filtered_gdf) == 0:
        st.warning("No regions match the selected filter criteria.")
        return
    
    # Create and display map
    st.markdown(f"**Displaying {len(filtered_gdf)} regions**")
    map_obj = create_map(filtered_gdf, filtered_predictions)
    
    # Display map
    map_data = st_folium(map_obj, width=1200, height=600)
    
    # Show selected region info if clicked
    if map_data['last_object_clicked_popup']:
        st.info("Click on a region on the map to see detailed information in the popup.")

def show_analytics(gdf: gpd.GeoDataFrame, predictions: pd.Series) -> None:  # type: ignore
    """Analytics and insights view"""
    st.markdown("## 📈 Advanced Analytics")
    
    # Correlation analysis
    st.markdown("### Feature Correlation with Priority Score")
    
    numeric_cols = ['population_density', 'health_facility_density', 'vulnerability_index',
                   'flooded_population', 'alluvial_plain_pct', 'base_flood_risk']
    
    if 'river_risk_score' in gdf.columns:
        numeric_cols.append('river_risk_score')
    if 'soil_saturation' in gdf.columns:
        numeric_cols.append('soil_saturation')
    
    # Calculate correlations
    correlations = {}
    for col in numeric_cols:
        if col in gdf.columns:
            corr = gdf[col].corr(predictions)
            if pd.notna(corr):
                correlations[col] = corr
    
    corr_df = pd.DataFrame({
        'Feature': list(correlations.keys()),
        'Correlation': list(correlations.values())
    }).sort_values('Correlation', key=abs, ascending=False)
    
    fig = px.bar(
        corr_df,
        x='Correlation',
        y='Feature',
        orientation='h',
        labels={'Correlation': 'Correlation with Priority Score'},
        color='Correlation',
        color_continuous_scale='RdBu',
        color_continuous_midpoint=0
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature distributions
    st.markdown("### Feature Distributions")
    
    selected_feature = st.selectbox(
        "Select Feature to Analyze",
        numeric_cols,
        index=0
    )
    
    if selected_feature in gdf.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(
                gdf,
                y=selected_feature,
                title=f"{selected_feature.replace('_', ' ').title()} Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                gdf,
                x=selected_feature,
                y='Priority_Score_Normalized',
                hover_data=['ADM1_EN', 'NAME_1'],
                title=f"{selected_feature.replace('_', ' ').title()} vs Priority Score",
                labels={selected_feature: selected_feature.replace('_', ' ').title(),
                       'Priority_Score_Normalized': 'Priority Score (Normalized)'}
            )
            st.plotly_chart(fig, use_container_width=True)

def show_about() -> None:
    """About page"""
    st.markdown("## ℹ️ About GeoRelief-AI")
    
    st.markdown("""
    ### Mission
    GeoRelief-AI is a scalable, modular AI platform designed to optimize humanitarian resource 
    allocation for any disaster, anywhere in the world.
    
    ### Key Features
    - **Multi-Layer Intelligence Model**: Integrates Hazard, Exposure, and Vulnerability layers
    - **Real-time Data Integration**: GloFAS hydrology, ERA5-Land soil data, Sentinel-1 imagery
    - **Social Vulnerability Mapping**: WPDX water access, IDMC displacement data
    - **Dynamic Priority Scoring**: Weighted formula accounting for multiple risk factors
    
    ### Priority Score Formula
    ```
    Priority = (Hazard × 0.6) + (Log(Pop) × 0.2) + (Vulnerability × 0.2)
    
    Where Hazard = (0.35 × RealTime Flood) + (0.30 × River Discharge) + 
                   (0.20 × Soil Saturation) + (0.15 × Hist. Floodplain)
    ```
    
    ### Technology Stack
    - **Backend**: Python, TensorFlow, Scikit-learn
    - **Geospatial**: GeoPandas, Rasterio, Shapely
    - **Visualization**: Streamlit, Folium, Plotly
    - **Data Sources**: WorldPop, GADM, HDX, Open-Meteo, UNOSAT, NGA
    
    ### SDG Alignment
    This project directly supports UN Sustainable Development Goals:
    - **SDG 11**: Sustainable Cities and Communities
    - **SDG 13**: Climate Action
    - **SDG 6**: Clean Water and Sanitation
    - **SDG 3**: Good Health and Well-being
    
    ### License
    Distributed under the MIT License.
    """)

if __name__ == "__main__":
    main()

