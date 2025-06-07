import warnings
import os

# Comprehensive warning suppression
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Now import everything else
import streamlit as st
import ee
import geemap.foliumap as geemap
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import google.generativeai as genai
from google.oauth2 import service_account
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import tempfile
import zipfile
import geopandas as gpd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google AI
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Page configuration
st.set_page_config(
    page_title="🛰️ Advanced Sentinel-2 Intelligence Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with more visual appeal
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        color: white;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
    }
    
    .category-header {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
        font-size: 1.2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        border-left: 6px solid #4facfe;
        margin: 1rem 0;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    .smart-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .recommendation-box {
        background: linear-gradient(135deg, #e8f5e8 0%, #d4f1d4 100%);
        padding: 2.5rem;
        border-radius: 15px;
        border-left: 6px solid #2e8b57;
        margin: 1.5rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    .index-description {
        background: rgba(255,255,255,0.9);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .trend-indicator {
        font-size: 1.4rem;
        font-weight: 700;
    }
    
    .trend-up { color: #28a745; }
    .trend-down { color: #dc3545; }
    .trend-stable { color: #ffc107; }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .stat-item {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border-top: 3px solid #4facfe;
    }
    
    .correlation-heatmap {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Google Earth Engine Authentication
@st.cache_resource
def ee_authenticate():
    try:
        json_creds = None
        if "json_key" in st.secrets:
            st.info("Authenticating with Google Earth Engine using service account...")
            json_creds = st.secrets["json_key"]
        if json_creds is not None:
            if isinstance(json_creds, (dict, st.runtime.secrets.AttrDict)):
                service_account_info = dict(json_creds)
            elif isinstance(json_creds, str):
                service_account_info = json.loads(json_creds)
            else:
                raise ValueError("Invalid json_key format in secrets. Expected dict, AttrDict, or JSON string.")
            if "client_email" not in service_account_info:
                raise ValueError("Service account email address missing in json_key")
            creds = service_account.Credentials.from_service_account_info(
                service_account_info, scopes=['https://www.googleapis.com/auth/earthengine']
            )
            ee.Initialize(creds)
            st.success("Successfully authenticated with Google Earth Engine using service account.")
        else:
            st.info("Attempting authentication using Earth Engine CLI credentials...")
            ee.Initialize()
            st.success("Authenticated with Google Earth Engine using local CLI credentials.")
    except Exception as e:
        st.error(f"Failed to authenticate with Google Earth Engine: {str(e)}")
        st.markdown(
            "**Steps to resolve:**\n"
            "- **Local setup**: Create `.streamlit/secrets.toml` with a valid service account key, or run `earthengine authenticate`.\n"
            "- **Cloud deployment**: Configure `[json_key]` in Streamlit secrets.\n"
            "- Ensure the service account has Earth Engine permissions (`roles/earthengine.user`).\n"
            "- Register at https://developers.google.com/earth-engine/guides/access."
        )
        st.stop()

# Initialize Earth Engine
ee_authenticate()

# Enhanced Comprehensive Sentinel-2 Indices Dictionary with Mining & Industrial Indices
INDICES_CATEGORIES = {
    '🌿 Vegetation Health': {
        'NDVI': {
            'formula': '(NIR - Red) / (NIR + Red)',
            'description': 'Primary vegetation health indicator - most widely used',
            'range': [-1, 1],
            'interpretation': 'Higher values (0.3-0.8) indicate healthier vegetation',
            'threshold': {'poor': 0.2, 'moderate': 0.4, 'good': 0.6, 'excellent': 0.8},
            'color': '#228B22'
        },
        'EVI': {
            'formula': '2.5 * ((NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1))',
            'description': 'Enhanced vegetation index - reduces atmospheric interference',
            'range': [-1, 1],
            'interpretation': 'Better for dense vegetation areas',
            'threshold': {'poor': 0.15, 'moderate': 0.3, 'good': 0.45, 'excellent': 0.6},
            'color': '#32CD32'
        },
        'SAVI': {
            'formula': '((NIR - Red) / (NIR + Red + 0.5)) * 1.5',
            'description': 'Soil-adjusted vegetation index - minimizes soil background',
            'range': [-1.5, 1.5],
            'interpretation': 'Ideal for sparse vegetation monitoring',
            'threshold': {'poor': 0.1, 'moderate': 0.25, 'good': 0.4, 'excellent': 0.6},
            'color': '#90EE90'
        },
        'GNDVI': {
            'formula': '(NIR - Green) / (NIR + Green)',
            'description': 'Green-based vegetation index - sensitive to chlorophyll',
            'range': [-1, 1],
            'interpretation': 'Excellent for crop monitoring and stress detection',
            'threshold': {'poor': 0.2, 'moderate': 0.35, 'good': 0.5, 'excellent': 0.65},
            'color': '#00FF00'
        },
        'NDRE': {
            'formula': '(NIR - RedEdge) / (NIR + RedEdge)',
            'description': 'Red-edge vegetation index - early stress detection',
            'range': [-1, 1],
            'interpretation': 'Highly sensitive to chlorophyll variations',
            'threshold': {'poor': 0.1, 'moderate': 0.2, 'good': 0.3, 'excellent': 0.4},
            'color': '#ADFF2F'
        },
        'ARVI': {
            'formula': '(NIR - (2*Red - Blue)) / (NIR + (2*Red - Blue))',
            'description': 'Atmospherically Resistant Vegetation Index - atmospheric correction',
            'range': [-1, 1],
            'interpretation': 'Reduces atmospheric effects on vegetation monitoring',
            'threshold': {'poor': 0.15, 'moderate': 0.3, 'good': 0.5, 'excellent': 0.7},
            'color': '#9ACD32'
        }
    },
    
    '💧 Water Resources': {
        'NDWI': {
            'formula': '(Green - NIR) / (Green + NIR)',
            'description': 'Primary water detection index for open water bodies',
            'range': [-1, 1],
            'interpretation': 'Values > 0.3 typically indicate water presence',
            'threshold': {'dry': -0.3, 'moist': 0.0, 'wet': 0.3, 'water': 0.5},
            'color': '#0000FF'
        },
        'MNDWI': {
            'formula': '(Green - SWIR1) / (Green + SWIR1)',
            'description': 'Modified water index - better for built-up areas',
            'range': [-1, 1],
            'interpretation': 'Enhanced water detection in urban environments',
            'threshold': {'dry': -0.2, 'moist': 0.1, 'wet': 0.4, 'water': 0.6},
            'color': '#1E90FF'
        },
        'AWEIsh': {
            'formula': 'Blue + 2.5*Green - 1.5*(NIR + SWIR1) - 0.25*SWIR2',
            'description': 'Automated Water Extraction Index (shadow) - handles shadows',
            'range': [-2, 2],
            'interpretation': 'Positive values indicate water with shadow tolerance',
            'threshold': {'no_water': -0.5, 'possible': 0.0, 'water': 0.5, 'certain': 1.0},
            'color': '#4169E1'
        },
        'AWEInsh': {
            'formula': '4*(Green - SWIR1) - (0.25*NIR + 2.75*SWIR2)',
            'description': 'Automated Water Extraction Index (no shadow) - pure water',
            'range': [-2, 2],
            'interpretation': 'Optimized for clear water detection',
            'threshold': {'no_water': -0.3, 'possible': 0.0, 'water': 0.3, 'certain': 0.8},
            'color': '#00BFFF'
        }
    },
    
    '🏙️ Urban Development': {
        'NDBI': {
            'formula': '(SWIR1 - NIR) / (SWIR1 + NIR)',
            'description': 'Built-up area identification index',
            'range': [-1, 1],
            'interpretation': 'Higher values indicate more built-up areas',
            'threshold': {'natural': -0.1, 'mixed': 0.0, 'urban': 0.1, 'dense_urban': 0.2},
            'color': '#FF4500'
        },
        'UI': {
            'formula': '(SWIR2 - NIR) / (SWIR2 + NIR)',
            'description': 'Urban Index - alternative built-up area detection',
            'range': [-1, 1],
            'interpretation': 'Complementary to NDBI for urban mapping',
            'threshold': {'natural': -0.05, 'mixed': 0.05, 'urban': 0.15, 'dense': 0.25},
            'color': '#FF6347'
        },
        'BUI': {
            'formula': '(NDVI - NDWI) / (NDVI + NDWI)',
            'description': 'Built-up Index - uses vegetation and water contrast',
            'range': [-1, 1],
            'interpretation': 'Effective for urban expansion monitoring',
            'threshold': {'natural': -0.2, 'mixed': 0.0, 'urban': 0.2, 'dense': 0.4},
            'color': '#CD5C5C'
        }
    },
    
    '🔥 Fire & Burn Analysis': {
        'NBR': {
            'formula': '(NIR - SWIR2) / (NIR + SWIR2)',
            'description': 'Normalized burn ratio for fire damage assessment',
            'range': [-1, 1],
            'interpretation': 'Lower values indicate burn damage',
            'threshold': {'severe_burn': -0.25, 'moderate_burn': 0.0, 'low_burn': 0.1, 'unburned': 0.3},
            'color': '#DC143C'
        },
        'NBRT': {
            'formula': '(NIR - SWIR2*T) / (NIR + SWIR2*T)',
            'description': 'NBR with thermal component - enhanced burn detection',
            'range': [-1, 1],
            'interpretation': 'Improved burn severity assessment',
            'threshold': {'severe': -0.3, 'moderate': -0.1, 'low': 0.1, 'unburned': 0.3},
            'color': '#B22222'
        },
        'BAI': {
            'formula': '1 / ((0.1 - Red)^2 + (0.06 - NIR)^2)',
            'description': 'Burn Area Index - highlights burned areas',
            'range': [0, 500],
            'interpretation': 'Higher values indicate burned areas',
            'threshold': {'unburned': 50, 'low_burn': 100, 'moderate': 200, 'severe': 300},
            'color': '#8B0000'
        }
    },
    
    '⛏️ Mining & Geology': {
        'FerricOxide': {
            'formula': 'SWIR1 / NIR',
            'description': 'Ferric Oxide detection - iron-rich minerals and mining areas',
            'range': [0, 5],
            'interpretation': 'Higher values indicate iron oxide presence (mining activity)',
            'threshold': {'low': 1.0, 'moderate': 1.5, 'high': 2.0, 'very_high': 3.0},
            'color': '#8B4513'
        },
        'ClayMinerals': {
            'formula': 'SWIR1 / SWIR2',
            'description': 'Clay mineral detection - alteration zones and tailings',
            'range': [0, 3],
            'interpretation': 'Detects clay alteration common in mining areas',
            'threshold': {'background': 0.8, 'weak': 1.0, 'moderate': 1.2, 'strong': 1.5},
            'color': '#D2691E'
        },
        'NDII': {
            'formula': '(NIR - SWIR1) / (NIR + SWIR1)',
            'description': 'Normalized Difference Infrared Index - moisture and mining impact',
            'range': [-1, 1],
            'interpretation': 'Detects moisture stress and bare soil from mining',
            'threshold': {'dry': -0.2, 'moderate': 0.0, 'moist': 0.2, 'wet': 0.4},
            'color': '#A0522D'
        },
        'SINDRI': {
            'formula': '(SWIR2 - Blue) / (SWIR2 + Blue)',
            'description': 'SWIR-NIR Difference Ratio Index - exposed soil and rocks',
            'range': [-1, 1],
            'interpretation': 'Highlights exposed surfaces typical of mining operations',
            'threshold': {'vegetated': -0.1, 'mixed': 0.1, 'exposed': 0.3, 'mining': 0.5},
            'color': '#CD853F'
        },
        'AlOH': {
            'formula': '(SWIR2 / SWIR1) * (SWIR1 / NIR)',
            'description': 'Aluminum Hydroxide - alteration mineral detection',
            'range': [0, 5],
            'interpretation': 'Detects hydrothermal alteration zones around mines',
            'threshold': {'background': 0.8, 'weak': 1.2, 'moderate': 1.8, 'strong': 2.5},
            'color': '#DEB887'
        }
    },
    
    '🏭 Industrial & Pollution': {
        'NDSI': {
            'formula': '(Green - SWIR1) / (Green + SWIR1)',
            'description': 'Normalized Difference Salinity Index - soil salination',
            'range': [-1, 1],
            'interpretation': 'Detects salt accumulation from industrial activities',
            'threshold': {'normal': -0.1, 'slight': 0.1, 'moderate': 0.3, 'severe': 0.5},
            'color': '#F5DEB3'
        },
        'NDTI': {
            'formula': '(Red - Green) / (Red + Green)',
            'description': 'Normalized Difference Turbidity Index - water pollution',
            'range': [-1, 1],
            'interpretation': 'Monitors water turbidity from industrial discharge',
            'threshold': {'clear': -0.1, 'slight': 0.1, 'turbid': 0.3, 'polluted': 0.5},
            'color': '#778899'
        },
        'APRI': {
            'formula': '(Blue - 2*Green + Red) / (Blue + 2*Green + Red)',
            'description': 'Atmospheric Pollution Ratio Index - air quality assessment',
            'range': [-1, 1],
            'interpretation': 'Detects atmospheric pollution from industrial sources',
            'threshold': {'clean': -0.1, 'moderate': 0.0, 'polluted': 0.2, 'severe': 0.4},
            'color': '#696969'
        }
    },
    
    '🌾 Agriculture Enhanced': {
        'MSAVI': {
            'formula': '(2*NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR - Red))) / 2',
            'description': 'Modified Soil Adjusted Vegetation Index - precision agriculture',
            'range': [-1, 1],
            'interpretation': 'Advanced vegetation monitoring with soil correction',
            'threshold': {'poor': 0.1, 'fair': 0.3, 'good': 0.5, 'excellent': 0.7},
            'color': '#32CD32'
        },
        'VARI': {
            'formula': '(Green - Red) / (Green + Red - Blue)',
            'description': 'Visible Atmospherically Resistant Index - crop stress',
            'range': [-1, 1],
            'interpretation': 'Resistant to atmospheric effects for crop monitoring',
            'threshold': {'stressed': 0.0, 'moderate': 0.2, 'healthy': 0.4, 'vigorous': 0.6},
            'color': '#228B22'
        },
        'CIG': {
            'formula': '(NIR / Green) - 1',
            'description': 'Chlorophyll Index Green - chlorophyll content estimation',
            'range': [0, 10],
            'interpretation': 'Direct chlorophyll content assessment',
            'threshold': {'low': 1.0, 'moderate': 2.0, 'good': 4.0, 'high': 6.0},
            'color': '#00FF00'
        }
    },
    
    '🌍 Soil & Geology': {
        'BSI': {
            'formula': '((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))',
            'description': 'Bare Soil Index - exposed soil detection',
            'range': [-1, 1],
            'interpretation': 'Identifies bare soil and erosion patterns',
            'threshold': {'vegetated': -0.2, 'mixed': 0.0, 'bare': 0.3, 'eroded': 0.6},
            'color': '#D2691E'
        },
        'GI': {
            'formula': 'Green / Red',
            'description': 'Greenness Index - vegetation greenness assessment',
            'range': [0, 5],
            'interpretation': 'Simple but effective vegetation greenness measure',
            'threshold': {'brown': 0.8, 'yellow': 1.0, 'light_green': 1.2, 'green': 1.5},
            'color': '#90EE90'
        },
        'RNDVI': {
            'formula': '(NIR - Red) / sqrt(NIR + Red)',
            'description': 'Renormalized Difference Vegetation Index - improved sensitivity',
            'range': [-1, 1],
            'interpretation': 'Enhanced vegetation detection in low biomass areas',
            'threshold': {'sparse': 0.1, 'moderate': 0.3, 'dense': 0.5, 'very_dense': 0.7},
            'color': '#228B22'
        }
    }
}

def calculate_comprehensive_indices(image):
    """Enhanced indices calculation with mining and industrial indices"""
    try:
        # Basic bands with null handling
        B2 = image.select('B2').unmask(0)   # Blue
        B3 = image.select('B3').unmask(0)   # Green
        B4 = image.select('B4').unmask(0)   # Red
        B5 = image.select('B5').unmask(0)   # Red Edge 1
        B8 = image.select('B8').unmask(0)   # NIR
        B11 = image.select('B11').unmask(0) # SWIR1
        B12 = image.select('B12').unmask(0) # SWIR2
        
        # Vegetation Indices
        ndvi = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI')
        evi = image.expression(
            '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
            {'NIR': B8, 'RED': B4, 'BLUE': B2}
        ).rename('EVI')
        savi = image.expression(
            '((NIR - RED) / (NIR + RED + 0.5)) * 1.5', 
            {'NIR': B8, 'RED': B4}
        ).rename('SAVI')
        gndvi = B8.subtract(B3).divide(B8.add(B3)).rename('GNDVI')
        ndre = B8.subtract(B5).divide(B8.add(B5)).rename('NDRE')
        arvi = image.expression(
            '(NIR - (2*RED - BLUE)) / (NIR + (2*RED - BLUE))',
            {'NIR': B8, 'RED': B4, 'BLUE': B2}
        ).rename('ARVI')
        
        # Water Indices
        ndwi = B3.subtract(B8).divide(B3.add(B8)).rename('NDWI')
        mndwi = B3.subtract(B11).divide(B3.add(B11)).rename('MNDWI')
        aweish = image.expression(
            'BLUE + 2.5*GREEN - 1.5*(NIR + SWIR1) - 0.25*SWIR2',
            {'BLUE': B2, 'GREEN': B3, 'NIR': B8, 'SWIR1': B11, 'SWIR2': B12}
        ).rename('AWEIsh')
        aweinsh = image.expression(
            '4*(GREEN - SWIR1) - (0.25*NIR + 2.75*SWIR2)',
            {'GREEN': B3, 'NIR': B8, 'SWIR1': B11, 'SWIR2': B12}
        ).rename('AWEInsh')
        
        # Urban/Built-up Indices
        ndbi = B11.subtract(B8).divide(B11.add(B8)).rename('NDBI')
        ui = B12.subtract(B8).divide(B12.add(B8)).rename('UI')
        bui = ndvi.subtract(ndwi).divide(ndvi.add(ndwi)).rename('BUI')
        
        # Fire/Burn Indices
        nbr = B8.subtract(B12).divide(B8.add(B12)).rename('NBR')
        nbrt = B8.subtract(B12).divide(B8.add(B12)).rename('NBRT')  # Simplified version
        bai = image.expression(
            '1 / ((0.1 - RED)*(0.1 - RED) + (0.06 - NIR)*(0.06 - NIR))',
            {'RED': B4, 'NIR': B8}
        ).rename('BAI')
        
        # Mining & Geology Indices
        ferric_oxide = B11.divide(B8).rename('FerricOxide')
        clay_minerals = B11.divide(B12).rename('ClayMinerals')
        ndii = B8.subtract(B11).divide(B8.add(B11)).rename('NDII')
        sindri = B12.subtract(B2).divide(B12.add(B2)).rename('SINDRI')
        aloh = B12.divide(B11).multiply(B11.divide(B8)).rename('AlOH')
        
        # Industrial & Pollution Indices
        ndsi = B3.subtract(B11).divide(B3.add(B11)).rename('NDSI')
        ndti = B4.subtract(B3).divide(B4.add(B3)).rename('NDTI')
        apri = image.expression(
            '(BLUE - 2*GREEN + RED) / (BLUE + 2*GREEN + RED)',
            {'BLUE': B2, 'GREEN': B3, 'RED': B4}
        ).rename('APRI')
        
        # Agriculture Enhanced Indices
        msavi = image.expression(
            '(2*NIR + 1 - sqrt((2*NIR + 1)*(2*NIR + 1) - 8*(NIR - RED))) / 2',
            {'NIR': B8, 'RED': B4}
        ).rename('MSAVI')
        vari = image.expression(
            '(GREEN - RED) / (GREEN + RED - BLUE)',
            {'GREEN': B3, 'RED': B4, 'BLUE': B2}
        ).rename('VARI')
        cig = B8.divide(B3).subtract(1).rename('CIG')
        
        # Soil & Geology Indices
        bsi = image.expression(
            '((SWIR1 + RED) - (NIR + BLUE)) / ((SWIR1 + RED) + (NIR + BLUE))',
            {'SWIR1': B11, 'RED': B4, 'NIR': B8, 'BLUE': B2}
        ).rename('BSI')
        gi = B3.divide(B4).rename('GI')
        rndvi = B8.subtract(B4).divide(B8.add(B4).sqrt()).rename('RNDVI')
        
        # Combine all indices
        all_indices = [
            # Vegetation
            ndvi, evi, savi, gndvi, ndre, arvi,
            # Water
            ndwi, mndwi, aweish, aweinsh,
            # Urban
            ndbi, ui, bui,
            # Fire
            nbr, nbrt, bai,
            # Mining
            ferric_oxide, clay_minerals, ndii, sindri, aloh,
            # Industrial
            ndsi, ndti, apri,
            # Agriculture
            msavi, vari, cig,
            # Soil
            bsi, gi, rndvi
        ]
        
        indices_image = image
        for index in all_indices:
            indices_image = indices_image.addBands(index)
        
        return indices_image
    except Exception as e:
        st.error(f"Error calculating indices: {str(e)}")
        return image

def get_comprehensive_time_series(geometry, start_date, end_date, cloud_cover=20):
    """Enhanced time series with better filtering"""
    try:
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(geometry) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)) \
            .sort('system:time_start')
        
        # Check if collection is empty
        size = collection.size()
        if size.getInfo() == 0:
            st.warning("No images found for the specified criteria. Try expanding the date range or increasing cloud cover threshold.")
            return None
        
        indices_collection = collection.map(calculate_comprehensive_indices)
        
        def get_image_stats(image):
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=20,
                maxPixels=1e9,
                bestEffort=True
            )
            return ee.Feature(None, stats.set('date', image.date().format('YYYY-MM-dd')))
        
        time_series = indices_collection.map(get_image_stats)
        return ee.FeatureCollection(time_series)
    except Exception as e:
        st.error(f"Error in time series calculation: {str(e)}")
        return None

def get_smart_trend_analysis(data):
    """Advanced trend analysis with statistical insights"""
    if len(data) < 3:
        return {"trend": "Insufficient data", "confidence": 0, "slope": 0}

    x = np.arange(len(data))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, data.values)
    
    # Determine trend strength
    if abs(r_value) > 0.7:
        strength = "Strong"
    elif abs(r_value) > 0.4:
        strength = "Moderate"
    else:
        strength = "Weak"

    if slope > 0.01:
        direction = "Increasing"
        emoji = "📈"
        color = "green"
    elif slope < -0.01:
        direction = "Decreasing"
        emoji = "📉"
        color = "red"
    else:
        direction = "Stable"
        emoji = "➡️"
        color = "orange"

    return {
        "trend": f"{strength} {direction}",
        "emoji": emoji,
        "color": color,
        "slope": slope,
        "r_value": r_value,
        "p_value": p_value,
        "confidence": abs(r_value) * 100
    }

def create_advanced_visualizations(df, categories):
    """Create comprehensive visualizations"""
    
    # 1. Category-wise Time Series with Subplots
    st.subheader("📊 Category-wise Index Analysis")
    
    for category, indices in categories.items():
        available_indices = [idx for idx in indices.keys() if idx in df.columns]
        if available_indices:
            st.markdown(f"""
            <div class="category-header">
                {category} Analysis
            </div>
            """, unsafe_allow_html=True)
            
            # Create subplot for each category
            fig = make_subplots(
                rows=len(available_indices), cols=1,
                subplot_titles=[f"{idx} - {indices[idx]['description']}" for idx in available_indices],
                vertical_spacing=0.05
            )
            
            colors = px.colors.qualitative.Set3
            # colors = px.colors.qualitative.Set3
            for i, idx in enumerate(available_indices):
                data = df[idx].dropna()
                if len(data) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=df['date'],
                            y=df[idx],
                            name=idx,
                            line=dict(color=indices[idx]['color'], width=3),
                            mode='lines+markers',
                            hovertemplate=f'<b>{idx}</b><br>Date: %{{x}}<br>Value: %{{y:.3f}}<extra></extra>'
                        ),
                        row=i+1, col=1
                    )
                    
                    # Add threshold lines
                    thresholds = indices[idx].get('threshold', {})
                    for threshold_name, threshold_value in thresholds.items():
                        fig.add_hline(
                            y=threshold_value,
                            line_dash="dash",
                            line_color="gray",
                            opacity=0.5,
                            annotation_text=threshold_name,
                            row=i+1, col=1
                        )
            
            fig.update_layout(
                height=300 * len(available_indices),
                title_text=f"{category} Time Series Analysis",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # 2. Correlation Heatmap
    st.subheader("🔗 Inter-Index Correlation Analysis")
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 1:
        correlation_matrix = df[numeric_columns].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Index Correlation Matrix",
            height=600,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Smart Statistical Summary
    st.subheader("📈 Smart Statistical Insights")
    
    stats_data = []
    for category, indices in categories.items():
        for idx in indices.keys():
            if idx in df.columns:
                data = df[idx].dropna()
                if len(data) > 0:
                    trend_analysis = get_smart_trend_analysis(data)
                    stats_data.append({
                        'Category': category,
                        'Index': idx,
                        'Current': data.iloc[-1],
                        'Mean': data.mean(),
                        'Std': data.std(),
                        'Trend': trend_analysis['trend'],
                        'Confidence': f"{trend_analysis['confidence']:.1f}",
                        'Emoji': trend_analysis['emoji']
                    })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        
        # Display as interactive table
        st.dataframe(
            stats_df.style.format({
                'Current': '{:.3f}',
                'Mean': '{:.3f}',
                'Std': '{:.3f}'
            }),
            use_container_width=True
        )

def create_smart_dashboard_metrics(df, categories):
    """Create smart dashboard with key metrics"""
    st.subheader("🎯 Smart Dashboard Metrics")
    
    # Calculate overall health score
    health_scores = {}
    for category, indices in categories.items():
        category_scores = []
        for idx in indices.keys():
            if idx in df.columns:
                data = df[idx].dropna()
                if len(data) > 0:
                    current_value = data.iloc[-1]
                    thresholds = indices[idx].get('threshold', {})
                    
                    # Calculate normalized score based on thresholds
                    if thresholds:
                        max_threshold = max(thresholds.values())
                        min_threshold = min(thresholds.values())
                        if max_threshold != min_threshold:
                            normalized_score = (current_value - min_threshold) / (max_threshold - min_threshold)
                            category_scores.append(max(0, min(1, normalized_score)) * 100)
        
        if category_scores:
            health_scores[category] = np.mean(category_scores)
    
    # Display health scores
    if health_scores:
        cols = st.columns(len(health_scores))
        for i, (category, score) in enumerate(health_scores.items()):
            with cols[i]:
                # Determine color based on score
                if score >= 70:
                    color = "green"
                    status = "Excellent"
                elif score >= 50:
                    color = "orange"
                    status = "Good"
                else:
                    color = "red"
                    status = "Needs Attention"
                
                st.metric(
                    label=f"{category} Health",
                    value=f"{score:.1f}%",
                    delta=status
                )

@st.cache_data(ttl=3600)
def get_enhanced_ai_recommendations(indices_data, location_name, categories):
    """Enhanced AI recommendations with deeper analysis"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prepare comprehensive data summary
        analysis_summary = f"""
        COMPREHENSIVE SATELLITE ANALYSIS REPORT
        ========================================
        Location: {location_name}
        Analysis Period: {indices_data['date'].min()} to {indices_data['date'].max()}
        Total Observations: {len(indices_data)}
        
        DETAILED INDEX ANALYSIS BY CATEGORY:
        """
        
        for category, indices in categories.items():
            analysis_summary += f"\n\n{category.upper()}:\n"
            analysis_summary += "=" * 50 + "\n"
            
            for idx in indices.keys():
                if idx in indices_data.columns:
                    data = indices_data[idx].dropna()
                    if len(data) > 0:
                        trend_analysis = get_smart_trend_analysis(data)
                        
                        analysis_summary += f"""
                        {idx} ({indices[idx]['description']}):
                        - Current Value: {data.iloc[-1]:.3f}
                        - Mean Value: {data.mean():.3f}
                        - Standard Deviation: {data.std():.3f}
                        - Trend: {trend_analysis['trend']} (Confidence: {trend_analysis['confidence']:.1f}%)
                        - Range: {indices[idx]['range']}
                        - Interpretation: {indices[idx]['interpretation']}
                        """
        
        prompt = f"""
        As a senior remote sensing and agricultural expert with 20+ years of experience, 
        provide a comprehensive analysis based on this Sentinel-2 satellite data:

        {analysis_summary}

        Please provide a detailed report with the following sections:

        1. EXECUTIVE SUMMARY
           - Overall assessment of the monitored area
           - Key findings and critical insights

        2. VEGETATION HEALTH ANALYSIS
           - Detailed vegetation condition assessment
           - Seasonal trends and patterns
           - Areas of concern or improvement

        3. WATER RESOURCES ASSESSMENT
           - Water availability and distribution
           - Moisture stress indicators
           - Irrigation recommendations

        4. ENVIRONMENTAL MONITORING
           - Urban development impacts
           - Fire risk assessment
           - Ecosystem health indicators

        5. ACTIONABLE RECOMMENDATIONS
           - Immediate actions required
           - Medium-term strategies
           - Long-term planning suggestions

        6. PREDICTIVE INSIGHTS
           - Expected trends based on current data
           - Risk factors to monitor
           - Opportunity areas

        Format your response with clear headings, bullet points, and specific numerical references to the data.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Unable to generate enhanced AI recommendations: {str(e)}"

# Enhanced Main Dashboard Function
def main():
    # Enhanced Header with better styling
    st.markdown("""
    <div class="main-header">
        <h1>🛰️ Advanced Sentinel-2 Intelligence Dashboard</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Next-Generation Satellite Imagery Analysis with AI-Powered Intelligence
        </p>
        <p style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem;">
            Comprehensive Multi-Spectral Index Analysis | Smart Trend Detection | Predictive Insights
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar with better organization
    st.sidebar.title("🔧 Analysis Configuration")
    st.sidebar.markdown("---")
    
    # Location selection with simplified options
    st.sidebar.markdown("### 🌍 Location Selection")
    location_option = st.sidebar.selectbox(
        "Choose Location Method:",
        ["📍 Custom Coordinates", "🏛️ Predefined Locations", "📄 Upload Shapefile"],
        help="Select how you want to define your area of interest"
    )

    # Enhanced Data Configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📅 Analysis Parameters")
    
    # Date range with better defaults
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", 
            value=datetime(2023, 1, 1),
            help="Start date for analysis"
        )
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=datetime(2023, 12, 31),
            help="End date for analysis"
        )
    
    cloud_cover = st.sidebar.slider(
        "Max Cloud Cover (%)", 
        0, 100, 20, 5,
        help="Maximum allowed cloud cover percentage"
    )

    # Index Selection - Available from the beginning
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Select Indices for Analysis")
    
    selected_indices = {}
    for category, indices in INDICES_CATEGORIES.items():
        st.sidebar.markdown(f"**{category}**")
        category_indices = {}
        for idx in indices.keys():
            category_indices[idx] = st.sidebar.checkbox(
                f"{idx}",
                value=(idx in ['NDVI', 'NDWI', 'NDBI']),  # Default selections
                help=f"{indices[idx]['description']}"
            )
        selected_indices.update(category_indices)
    
    # Filter to get only selected indices
    indices_to_analyze = [idx for idx, selected in selected_indices.items() if selected]
    
    if not indices_to_analyze:
        st.sidebar.warning("⚠️ Please select at least one index for analysis")

    geometry = None
    location_name = ""
    lat, lon = 28.6139, 77.2090  # Default coordinates

    if location_option == "📍 Custom Coordinates":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Enter Coordinates:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=28.6139, format="%.6f", help="Enter latitude in decimal degrees")
        with col2:
            lon = st.number_input("Longitude", value=77.2090, format="%.6f", help="Enter longitude in decimal degrees")
        
        buffer_size = st.sidebar.slider(
            "Buffer Size (km)", 
            0.5, 100.0, 5.0, 0.5,
            help="Size of the analysis area around the point"
        )
        geometry = ee.Geometry.Point([lon, lat]).buffer(buffer_size * 1000)
        location_name = f"Custom Location ({lat:.4f}, {lon:.4f})"

        # Show selected indices info for custom coordinates
        if indices_to_analyze:
            st.sidebar.success(f"✅ {len(indices_to_analyze)} indices selected for analysis")

    elif location_option == "🏛️ Predefined Locations":
        st.sidebar.markdown("---")
        locations = {
            "New Delhi, India": [77.2090, 28.6139],
            "Punjab Agricultural Region, India": [75.3412, 31.1471],
            "California Central Valley, USA": [-120.4179, 36.7783],
            "Nile Delta, Egypt": [31.2357, 30.8025],
            "Amazon Basin, Brazil": [-60.0261, -3.4653],
            "Sahara Desert, Algeria": [1.6596, 28.0339],
        }
        selected_location = st.sidebar.selectbox("🌍 Choose Location", list(locations.keys()))
        lon, lat = locations[selected_location]
        buffer_size = st.sidebar.slider("Buffer Size (km)", 1.0, 50.0, 10.0, 1.0)
        geometry = ee.Geometry.Point([lon, lat]).buffer(buffer_size * 1000)
        location_name = selected_location

        # Show selected indices info for predefined locations
        if indices_to_analyze:
            st.sidebar.success(f"✅ {len(indices_to_analyze)} indices selected for analysis")

    elif location_option == "📄 Upload Shapefile":
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Upload a zipped Shapefile (.zip):**")
        uploaded_file = st.sidebar.file_uploader(
            "Upload Shapefile", 
            type=["zip"],
            help="Upload a zipped shapefile containing .shp, .shx, .dbf files"
        )
        if uploaded_file is not None:
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    zip_path = os.path.join(tmpdir, "shapefile.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_file.read())
                    with zipfile.ZipFile(zip_path, "r") as zip_ref:
                        zip_ref.extractall(tmpdir)
                    
                    shp_files = [f for f in os.listdir(tmpdir) if f.endswith(".shp")]
                    if shp_files:
                        shp_path = os.path.join(tmpdir, shp_files[0])
                        gdf = gpd.read_file(shp_path)
                        if not gdf.empty:
                            geometry = geemap.geopandas_to_ee(gdf)
                            location_name = "Uploaded Shapefile"
                            st.sidebar.success("✅ Shapefile loaded successfully.")
                        else:
                            st.sidebar.warning("⚠️ Uploaded shapefile is empty.")
                    else:
                        st.sidebar.warning("⚠️ No .shp file found in the uploaded zip.")
            except Exception as e:
                st.sidebar.error(f"❌ Error reading shapefile: {str(e)}")

        # Show selected indices info for shapefile
        if indices_to_analyze:
            st.sidebar.success(f"✅ {len(indices_to_analyze)} indices selected for analysis")

    # Data Processing and Analysis
    if geometry is not None and indices_to_analyze:
        if start_date > end_date:
            st.error("❌ Start date must be before end date.")
        else:
            # Enhanced data loading with progress
            with st.spinner("🛰️ Loading Sentinel-2 data..."):
                try:
                    # Get comprehensive time series data
                    time_series = get_comprehensive_time_series(
                        geometry,
                        str(start_date),
                        str(end_date),
                        cloud_cover
                    )
                    
                    if time_series is not None:
                        df = geemap.ee_to_df(time_series)
                        
                        if df is not None and not df.empty:
                            st.success(f"✅ Successfully loaded {len(df)} observations!")
                            
                            # Data preprocessing
                            df['date'] = pd.to_datetime(df['date'])
                            df = df.sort_values('date').reset_index(drop=True)
                            
                            # Remove any completely null rows
                            df = df.dropna(how='all', subset=[col for col in df.columns if col != 'date'])
                            
                            # Filter dataframe to only include selected indices
                            columns_to_keep = ['date'] + [idx for idx in indices_to_analyze if idx in df.columns]
                            df_filtered = df[columns_to_keep].copy()
                            
                            # Store in session state for AI analysis
                            st.session_state['analysis_df'] = df_filtered
                            st.session_state['location_name'] = location_name
                            
                            # Show AOI information
                            if geometry:
                                try:
                                    area_info = geometry.area().getInfo()
                                    area_km2 = area_info / 1000000
                                    
                                    st.info(f"""
                                    📍 **Analysis Area Information**
                                    - **Method**: {location_option}
                                    - **Location**: {location_name}
                                    - **Area**: {area_km2:.2f} km²
                                    - **Selected Indices**: {', '.join(indices_to_analyze)}
                                    - **Observations**: {len(df_filtered)} satellite images processed
                                    - **Date Range**: {df_filtered['date'].min().strftime('%Y-%m-%d')} to {df_filtered['date'].max().strftime('%Y-%m-%d')}
                                    """)
                                except:
                                    st.info(f"📍 **Analysis Area**: {location_name} | **Observations**: {len(df_filtered)}")
                            
                            # Create smart dashboard metrics for selected indices only
                            filtered_categories = {}
                            for category, indices in INDICES_CATEGORIES.items():
                                filtered_indices = {k: v for k, v in indices.items() if k in indices_to_analyze}
                                if filtered_indices:
                                    filtered_categories[category] = filtered_indices
                            
                            create_smart_dashboard_metrics(df_filtered, filtered_categories)
                            
                            # Enhanced data display
                            with st.expander("📊 Raw Data Explorer", expanded=False):
                                st.markdown("### Data Overview")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Observations", len(df_filtered))
                                with col2:
                                    st.metric("Date Range", f"{(df_filtered['date'].max() - df_filtered['date'].min()).days} days")
                                with col3:
                                    st.metric("Selected Indices", len(indices_to_analyze))
                                
                                st.dataframe(df_filtered, use_container_width=True)
                            
                            # Create advanced visualizations for selected indices only
                            create_advanced_visualizations(df_filtered, filtered_categories)
                            
                            # Enhanced Map Visualization with Selected Indices Only
                            st.subheader("🗺️ Interactive Map Visualization")
                            
                            # PROPERLY INTEGRATED MAP SECTION
                            col1, col2 = st.columns([2, 1])
                            
                            with col2:
                                # Controls and legends moved to right column
                                st.markdown("### 🎛️ Map Controls")
                                
                                # Map style selection
                                map_style = st.selectbox(
                                    "Map Style",
                                    ["OpenStreetMap", "Satellite", "Terrain"],
                                    help="Choose the base map style"
                                )
                                
                                # Geometry display control
                                show_geometry = st.checkbox("Show Analysis Area", value=True)
                                
                                st.markdown("---")
                                st.markdown("### 📊 Index Layers")
                                
                                # Show only selected indices for map layers that exist in the data
                                map_layers = {}
                                available_indices = [idx for idx in indices_to_analyze if idx in df_filtered.columns and not df_filtered[idx].isna().all()]
                                
                                if available_indices:
                                    for idx in available_indices:
                                        map_layers[idx] = st.checkbox(
                                            f"{idx}",
                                            value=(idx == available_indices[0]),  # First available index by default
                                            help=f"Show {idx} layer on map"
                                        )
                                else:
                                    st.warning("No valid indices available for mapping")
                                
                                st.markdown("---")
                                st.markdown("### 🎨 Options")
                                
                                # Opacity control
                                layer_opacity = st.slider(
                                    "Layer Opacity",
                                    0.1, 1.0, 0.7, 0.1,
                                    help="Adjust transparency of index layers"
                                )
                                
                                # Date selection for visualization
                                if len(df_filtered) > 1:
                                    date_options = df_filtered['date'].dt.strftime('%Y-%m-%d').tolist()
                                    selected_date = st.selectbox(
                                        "Select Date",
                                        date_options,
                                        index=len(date_options)-1,  # Default to latest date
                                        help="Choose which date to visualize on the map"
                                    )
                                else:
                                    selected_date = df_filtered['date'].iloc[0].strftime('%Y-%m-%d')

                                # Color Palette Legends - COMPACT VERSION
                                st.markdown("---")
                                st.markdown("### 🎨 Legends")
                                
                                # Show legends only for active layers
                                active_layers = [idx for idx, is_selected in map_layers.items() if is_selected]
                                
                                if active_layers:
                                    for idx in active_layers:
                                        # Find the category and get information
                                        index_info = None
                                        for category, indices in INDICES_CATEGORIES.items():
                                            if idx in indices:
                                                index_info = indices[idx]
                                                break
                                        
                                        if index_info:
                                            # Compact legend display
                                            with st.expander(f"📊 {idx}", expanded=True):
                                                st.caption(f"{index_info['description']}")
                                                
                                                # Create color palette based on index type
                                                if idx in ['NDVI', 'EVI', 'SAVI', 'GNDVI', 'NDRE', 'ARVI', 'MSAVI', 'VARI', 'RNDVI']:
                                                    # Vegetation indices - green palette
                                                    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
                                                    values = ['-0.2', '-0.1', '0.0', '0.1', '0.3', '0.5', '0.7', '0.8+']
                                                    interpretations = ['Bare soil', 'Very poor', 'Poor', 'Sparse', 'Moderate', 'Good', 'Dense', 'Very dense']
                                                
                                                elif idx in ['NDWI', 'MNDWI', 'AWEIsh', 'AWEInsh']:
                                                    # Water indices - blue palette
                                                    colors = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac', '#01665e', '#003c30']
                                                    values = ['-0.5', '-0.3', '-0.1', '0.0', '0.2', '0.4', '0.5+']
                                                    interpretations = ['Very dry', 'Dry', 'Moist', 'Wet', 'Standing water', 'Water bodies', 'Deep water']
                                                
                                                elif idx in ['NDBI', 'UI', 'BUI']:
                                                    # Urban indices
                                                    colors = ['#2166ac', '#5aae61', '#a6dba0', '#d9f0d3', '#f7f7f7', '#e7d4e8', '#c2a5cf', '#9970ab']
                                                    values = ['-0.3', '-0.2', '-0.1', '0.0', '0.1', '0.2', '0.3', '0.4+']
                                                    interpretations = ['Water/Veg', 'Natural', 'Mixed', 'Transition', 'Light urban', 'Moderate', 'Dense', 'Very dense']
                                                
                                                elif idx in ['NBR', 'NBRT', 'BAI']:
                                                    # Fire/Burn indices
                                                    colors = ['#67001f', '#b2182b', '#d6604d', '#f4a582', '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3']
                                                    values = ['-0.5', '-0.3', '-0.1', '0.0', '0.1', '0.3', '0.5', '0.7+']
                                                    interpretations = ['Severe burn', 'High burn', 'Moderate burn', 'Low burn', 'Unburned', 'Healthy veg', 'Dense veg', 'Very dense']
                                                
                                                elif idx in ['FerricOxide', 'ClayMinerals', 'AlOH']:
                                                    # Mining indices
                                                    colors = ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', '#df65b0', '#e7298a', '#ce1256', '#91003f']
                                                    values = ['0.5', '0.8', '1.0', '1.5', '2.0', '2.5', '3.0', '4.0+']
                                                    interpretations = ['Background', 'Very low', 'Low mineral', 'Moderate', 'High mineral', 'Very high', 'Alteration', 'Strong alt']
                                                
                                                else:
                                                    # Default for other indices
                                                    colors = ['#d73027', '#f46d43', '#fdae61', '#fee08b', '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
                                                    values = ['Low', '', '', '', '', '', '', 'High']
                                                    interpretations = ['Low', '', '', '', '', '', '', 'High']
                                                
                                                # Create compact legend
                                                legend_html = ""
                                                for color, value, interp in zip(colors, values, interpretations):
                                                    if value and interp:
                                                        legend_html += f"""
                                                        <div style="display: flex; align-items: center; margin: 1px 0; font-size: 0.8em;">
                                                            <div style="width: 15px; height: 10px; background-color: {color}; margin-right: 5px; border: 1px solid #ccc;"></div>
                                                            <span style="font-weight: bold; min-width: 30px;">{value}</span>
                                                            <span style="margin-left: 5px;">{interp}</span>
                                                        </div>
                                                        """
                                                
                                                st.markdown(legend_html, unsafe_allow_html=True)
                                                
                                                # Range info
                                                st.markdown(f"**Range:** {index_info['range'][0]} to {index_info['range'][1]}")
                                else:
                                    st.info("Select layers to see legends")

                            with col1:
                                # Create the map - now properly integrated with data
                                try:
                                    # Get centroid for map center
                                    centroid = geometry.centroid().getInfo()['coordinates']
                                    map_center = [centroid[1], centroid[0]]  # lat, lon
                                except:
                                    map_center = [lat, lon]  # fallback
                                
                                # Create map with better sizing
                                Map = geemap.Map(center=map_center, zoom=12)
                                
                                # Add base layer based on selection
                                if map_style == "Satellite":
                                    Map.add_basemap('SATELLITE')
                                elif map_style == "Terrain":
                                    Map.add_basemap('TERRAIN')
                                # OpenStreetMap is default
                                
                                # Add analysis area if selected
                                if show_geometry:
                                    Map.addLayer(
                                        geometry, 
                                        {
                                            'color': '#FF0000',
                                            'fillColor': '#FF0000',
                                            'fillOpacity': 0.1,
                                            'weight': 2
                                        }, 
                                        'Analysis Area'
                                    )
                                
                                # Add selected index layers - THIS IS THE KEY FIX
                                try:
                                    if map_layers and any(map_layers.values()):  # Check if any layers are selected
                                        # Get the image for the selected date
                                        target_date = pd.to_datetime(selected_date)
                                        date_buffer = 5  # days
                                        
                                        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                                            .filterBounds(geometry) \
                                            .filterDate(
                                                (target_date - timedelta(days=date_buffer)).strftime('%Y-%m-%d'),
                                                (target_date + timedelta(days=date_buffer)).strftime('%Y-%m-%d')
                                            ) \
                                            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover)) \
                                            .sort('system:time_start') \
                                            .first()
                                        
                                        if collection:
                                            # Calculate indices for the selected image
                                            image_with_indices = calculate_comprehensive_indices(collection)
                                            
                                            # Add each selected map layer - CLIPPED TO ANALYSIS AREA
                                            for idx, is_selected in map_layers.items():
                                                if is_selected:
                                                    # Find the category and get color/visualization parameters
                                                    index_info = None
                                                    for category, indices_cat in INDICES_CATEGORIES.items():
                                                        if idx in indices_cat:
                                                            index_info = indices_cat[idx]
                                                            break
                                                    
                                                    if index_info:
                                                        # Define visualization parameters based on index
                                                        vis_params = {
                                                            'opacity': layer_opacity
                                                        }
                                                        
                                                        if idx in ['NDVI', 'EVI', 'SAVI', 'GNDVI', 'NDRE', 'ARVI', 'MSAVI', 'VARI', 'RNDVI']:
                                                            # Vegetation indices - green palette
                                                            vis_params.update({
                                                                'min': -0.2,
                                                                'max': 0.8,
                                                                'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', 
                                                                          '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
                                                            })
                                                        elif idx in ['NDWI', 'MNDWI', 'AWEIsh', 'AWEInsh']:
                                                            # Water indices - blue palette
                                                            vis_params.update({
                                                                'min': -0.5,
                                                                'max': 0.5,
                                                                'palette': ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', 
                                                                          '#5ab4ac', '#01665e', '#003c30']
                                                            })
                                                        elif idx in ['NDBI', 'UI', 'BUI']:
                                                            # Urban index - red/orange palette
                                                            vis_params.update({
                                                                'min': -0.3,
                                                                'max': 0.3,
                                                                'palette': ['#2166ac', '#5aae61', '#a6dba0', '#d9f0d3',
                                                                          '#f7f7f7', '#e7d4e8', '#c2a5cf', '#9970ab']
                                                            })
                                                        elif idx in ['NBR', 'NBRT', 'BAI']:
                                                            # Burn index - fire palette
                                                            vis_params.update({
                                                                'min': -0.5,
                                                                'max': 0.5,
                                                                'palette': ['#67001f', '#b2182b', '#d6604d', '#f4a582',
                                                                          '#fddbc7', '#d1e5f0', '#92c5de', '#4393c3']
                                                            })
                                                        elif idx in ['FerricOxide', 'ClayMinerals', 'AlOH']:
                                                            # Mining indices - purple/brown palette
                                                            vis_params.update({
                                                                'min': 0.5,
                                                                'max': 4.0,
                                                                'palette': ['#f7f4f9', '#e7e1ef', '#d4b9da', '#c994c7', 
                                                                          '#df65b0', '#e7298a', '#ce1256', '#91003f']
                                                            })
                                                        elif idx in ['NDII', 'SINDRI']:
                                                            # Moisture/soil indices - earth tones
                                                            vis_params.update({
                                                                'min': -0.4,
                                                                'max': 0.4,
                                                                'palette': ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', 
                                                                          '#c7eae5', '#80cdc1', '#35978f', '#01665e']
                                                            })
                                                        elif idx in ['NDSI', 'NDTI', 'APRI']:
                                                            # Pollution indices - blue to red
                                                            vis_params.update({
                                                                'min': -0.3,
                                                                'max': 0.4,
                                                                'palette': ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', 
                                                                          '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
                                                            })
                                                        elif idx in ['BSI', 'GI']:
                                                            # Soil/greenness indices
                                                            vis_params.update({
                                                                'min': -0.3,
                                                                'max': 0.4,
                                                                'palette': ['#762a83', '#9970ab', '#c2a5cf', '#e7d4e8', 
                                                                          '#d9ef8b', '#a6d96a', '#66bd63', '#1b7837']
                                                            })
                                                        elif idx == 'CIG':
                                                            # Chlorophyll index
                                                            vis_params.update({
                                                                'min': 0,
                                                                'max': 6,
                                                                'palette': ['#ffffcc', '#d9f0a3', '#addd8e', '#78c679', 
                                                                          '#41ab5d', '#238443', '#005a32']
                                                            })
                                                        else:
                                                            # Default palette
                                                            vis_params.update({
                                                                'min': index_info['range'][0],
                                                                'max': index_info['range'][1],
                                                                'palette': ['#d73027', '#f46d43', '#fdae61', '#fee08b', 
                                                                          '#d9ef8b', '#a6d96a', '#66bd63', '#1a9850']
                                                            })
                                                    
                                                    # CLIP THE INDEX LAYER TO THE ANALYSIS AREA
                                                    clipped_index = image_with_indices.select(idx).clip(geometry)
                                                    
                                                    # Add the clipped layer to the map
                                                    Map.addLayer(
                                                        clipped_index,
                                                        vis_params,
                                                        f'{idx} ({selected_date})'
                                                    )
                                                    
                                                    st.success(f"✅ Added {idx} layer to map")
                                        else:
                                            st.warning(f"No satellite images found for {selected_date}. Try selecting a different date or increasing the cloud cover threshold.")
                                    else:
                                        st.info("Select index layers in the sidebar to visualize them on the map")
                                        
                                except Exception as e:
                                    st.error(f"Error adding index layers: {str(e)}")
                                
                                # Add a border around the analysis area for clarity
                                if show_geometry:
                                    Map.addLayer(
                                        geometry, 
                                        {
                                            'color': '#FF0000',
                                            'fillColor': 'rgba(0,0,0,0)',  # Transparent fill
                                            'weight': 3,
                                            'dashArray': '5, 5'  # Dashed border
                                        }, 
                                        'Analysis Area Boundary'
                                    )
                                
                                # Display the map with better height
                                Map.to_streamlit(height=650)
                        
                        else:
                            st.error("❌ No data returned from Earth Engine. Please try different parameters.")
                    else:
                        st.error("❌ Failed to retrieve data from Earth Engine.")
                        
                except Exception as e:
                    st.error(f"Error loading Sentinel-2 data: {str(e)}")

    # Enhanced AI Recommendations Section (remains the same)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 AI Intelligence")
    
    if st.sidebar.button("🚀 Generate Smart Analysis", type="primary"):
        if 'analysis_df' in st.session_state and st.session_state['analysis_df'] is not None:
            with st.spinner("🧠 Generating AI-powered insights..."):
                try:
                    # Use filtered categories for AI analysis
                    filtered_categories = {}
                    if indices_to_analyze:
                        for category, indices in INDICES_CATEGORIES.items():
                            filtered_indices = {k: v for k, v in indices.items() if k in indices_to_analyze}
                            if filtered_indices:
                                filtered_categories[category] = filtered_indices
                    
                    recommendations = get_enhanced_ai_recommendations(
                        st.session_state['analysis_df'], 
                        st.session_state['location_name'], 
                        filtered_categories
                    )
                    
                    st.markdown(f"""
                    <div class="recommendation-box">
                        <h2>🤖 AI-Powered Intelligence Report</h2>
                        <h3>📍 Location: {st.session_state['location_name']}</h3>
                        <h4>📊 Analyzed Indices: {', '.join(indices_to_analyze)}</h4>
                        <div style="white-space: pre-wrap; font-family: 'Inter', sans-serif; line-height: 1.6;">
                        {recommendations}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ Error generating AI analysis: {str(e)}")
        else:
            st.warning("⚠️ Please load data first before generating AI analysis.")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 10px; margin-top: 1rem;">
    <h4>🛰️ About This Dashboard</h4>
    <p style="font-size: 0.85rem; color: #666;">
    Advanced satellite imagery analysis using Sentinel-2 data with AI-powered insights for vegetation, water, urban,fire monitoring, Mining & Geology & Soil & Geology .
    </p>
    <p style="font-size: 0.8rem; font-weight: bold; color: #333;">
    Developed by: Dr. Anil Kumar Singh
    </p>
    <p style="font-size: 0.8rem;">
        <a href="https://www.linkedin.com/in/anil-kumar-singh-phd-b192554a/" target="_blank" style="text-decoration: none;">
            <img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png" width="20" style="vertical-align: middle;"/> LinkedIn Profile
        </a>
    </p>
    <p style="font-size: 0.8rem;">
        <a href="mailto:singhanil854@gmail.com" style="text-decoration: none;">
            <img src="https://www.pngall.com/wp-content/uploads/12/Gmail-Logo-PNG-Images.png" width="20" style="vertical-align: middle;"/> singhanil854@gmail.com
        </a>
    </p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()