import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
import warnings
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# --- Streamlit Page Config ---
try:
    st.set_page_config(
        page_title="ZynfoGraph",
        page_icon="android-chrome-512x512.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Error setting page config: {str(e)}")

st.markdown(
    """
    <style>
    /* Floating animation */
    @keyframes float {
        0%, 100% {
            transform: translateY(0px);
            box-shadow: 0 0 15px 3px rgba(68, 68, 68, 0.6);
        }
        50% {
            transform: translateY(-10px);
            box-shadow: 0 0 25px 6px rgba(68, 68, 68, 0.9);
        }
    }
    .profile-logo {
        position: fixed;
        top: 40px;
        right: 45px;
        width: 90px;
        height: 90px;
        border-radius: 50%; /* Perfect circle */
        object-fit: cover;
        border: 2px solid #444;
        box-shadow: 0 0 15px 3px rgba(68, 68, 68, 0.6); /* Initial glow */
        animation: float 4s ease-in-out infinite; /* Floating animation */
        z-index: 100;
        transition: box-shadow 0.3s ease;
    }
    /* Optional: stronger glow on hover */
    .profile-logo:hover {
        box-shadow: 0 0 35px 10px rgba(68, 68, 68, 1);
        cursor: pointer;
    }
    </style>
    <img src="https://i.postimg.cc/nrL8ScR2/Zynfo-Graph-icon.jpg" class="profile-logo" alt="Profile Logo">
    """,
    unsafe_allow_html=True)

st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)

# --- Enhanced Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
        .main-header {
        background: linear-gradient(90deg, #00d4aa, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1rem;
    }
        .stButton>button {
        background: linear-gradient(45deg, #00b4d8, #00d4aa);
        color: white !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
        .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.3);
    }
        .metric-card {
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 1.5rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        margin: 1rem 0 !important;
        text-align: center !important;
        min-height: 120px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
        .metric-card h3 {
        color: #00d4aa !important;
        margin-bottom: 0.5rem !important;
        font-size: 1.1rem !important;
    }
        .metric-card p {
        color: rgba(255, 255, 255, 0.8) !important;
        font-size: 0.9rem !important;
        margin: 0 !important;
    }
        .error-container {
        background: rgba(255, 0, 0, 0.1) !important;
        border: 1px solid rgba(255, 0, 0, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
        .success-container {
        background: rgba(0, 255, 0, 0.1) !important;
        border: 1px solid rgba(0, 255, 0, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
        .stSelectbox>div>div>select {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 8px !important;
    }
        .stDataFrame {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
        /* Fix for HTML rendering issues and column alignment */
    div[data-testid="stMarkdownContainer"] {
        color: inherit !important;
        /* Ensure markdown content stretches vertically within columns */
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }
    </style>""", unsafe_allow_html=True)

# --- Enhanced Utility Functions ---
@st.cache_data
def load_data(uploaded_file_obj): # Renamed parameter to avoid confusion with session state
    """Enhanced data loading with comprehensive error handling"""
    try:
        if uploaded_file_obj is None:
            return None, "No file uploaded"
                    
        # Reset file pointer
        uploaded_file_obj.seek(0)
                    
        file_ext = uploaded_file_obj.name.split('.')[-1].lower()
                    
        if file_ext == 'csv':
            # Try different encodings and separators
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            separators = [',', ';', '\t']
                        
            for encoding in encodings:
                for sep in separators:
                    try:
                        uploaded_file_obj.seek(0)
                        df = pd.read_csv(uploaded_file_obj, encoding=encoding, sep=sep)
                        if df.shape[1] > 1: # Valid dataframe
                            return df, f"Successfully loaded the file..."
                    except Exception:
                        continue
            return None, "Failed to load CSV with any encoding/separator combination"
                    
        elif file_ext in ['xls', 'xlsx']:
            try:
                df = pd.read_excel(uploaded_file_obj)
                return df, f"Successfully loaded Excel file"
            except Exception as e:
                return None, f"Error loading Excel file: {str(e)}"
                    
        elif file_ext == 'json':
            try:
                uploaded_file_obj.seek(0)
                df = pd.read_json(uploaded_file_obj)
                return df, f"Successfully loaded JSON file"
            except Exception as e:
                return None, f"Error loading JSON file: {str(e)}"
                    
        elif file_ext in ['txt', 'tsv']:
            try:
                uploaded_file_obj.seek(0)
                df = pd.read_csv(uploaded_file_obj, delimiter='\t')
                return df, f"Successfully loaded TSV/TXT file"
            except Exception as e:
                return None, f"Error loading TSV/TXT file: {str(e)}"
                    
        elif file_ext == 'parquet':
            try:
                df = pd.read_parquet(uploaded_file_obj)
                return df, f"Successfully loaded Parquet file"
            except Exception as e:
                return None, f"Error loading Parquet file: {str(e)}"
        else:
            return None, f"Unsupported file type: {file_ext}"
                    
    except Exception as e:
        logger.error(f"Unexpected error in load_data: {str(e)}")
        return None, f"Unexpected error loading file: {str(e)}"

def safe_get_data_profile(df):
    """Generate comprehensive data profile with error handling"""
    try:
        profile = {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum(),
            'data_types': df.dtypes,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist()
        }
        return profile, None
    except Exception as e:
        return None, f"Error generating data profile: {str(e)}"

def safe_clean_data(df, options):
    """Apply data cleaning operations with error handling"""
    if df is None:
        return None, [], "Input DataFrame is None, cannot clean."
    try:
        cleaned_df = df.copy()
        operations_performed = []
                    
        if options.get('remove_duplicates', False):
            initial_rows = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_rows = initial_rows - len(cleaned_df)
            if removed_rows > 0:
                operations_performed.append(f"Removed {removed_rows} duplicate rows")
                        
        if options.get('handle_missing', 'keep') != 'keep':
            # missing_before = cleaned_df.isnull().sum().sum() # This line is not used, can remove
                        
            if options['handle_missing'] == 'drop':
                cleaned_df = cleaned_df.dropna()
                operations_performed.append(f"Dropped rows with missing values")
                            
            elif options['handle_missing'] == 'fill_mean':
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].mean())
                    operations_performed.append(f"Filled missing values with mean for {len(numeric_cols)} numeric columns")
                            
            elif options['handle_missing'] == 'fill_median':
                numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    cleaned_df[numeric_cols] = cleaned_df[numeric_cols].fillna(cleaned_df[numeric_cols].median())
                    operations_performed.append(f"Filled missing values with median for {len(numeric_cols)} numeric columns")
        return cleaned_df, operations_performed, None # Success case
                    
    except Exception as e:
        # If any error occurs during cleaning, return None for cleaned_df
        return None, [], f"Error during data cleaning: {str(e)}"

def export_plot_as_image(fig, format_type="png", filename="plot"):
    """Export plotly figure as PNG or PDF with error handling"""
    try:
        if format_type.lower() == "png":
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            return img_bytes, f"{filename}.png", "image/png", None
        elif format_type.lower() == "pdf":
            img_bytes = fig.to_image(format="pdf", width=1200, height=800)
            return img_bytes, f"{filename}.pdf", "application/pdf", None
        else:
            return None, None, None, f"Unsupported format: {format_type}"
    except Exception as e:
        # Fallback: try using matplotlib for simple plots
        try:
            import matplotlib.pyplot as plt
            from io import BytesIO
                        
            # Convert plotly to matplotlib (basic fallback)
            buffer = BytesIO()
            plt.figure(figsize=(12, 8))
            plt.text(0.5, 0.5, f"Plot export failed.\nOriginal error: {str(e)}", 
                             ha='center', va='center', transform=plt.gca().transAxes)
            plt.title("Export Error - Please try HTML export instead")
                        
            if format_type.lower() == "png":
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                mime_type = "image/png"
                ext = "png"
            else:
                plt.savefig(buffer, format='pdf', bbox_inches='tight')
                mime_type = "application/pdf"
                ext = "pdf"
                        
            plt.close()
            buffer.seek(0)
            return buffer.getvalue(), f"{filename}_fallback.{ext}", mime_type, f"Warning: Used fallback export due to error: {str(e)}"
                        
        except Exception as fallback_error:
            return None, None, None, f"Export failed: {str(e)}. Fallback also failed: {str(fallback_error)}"

def safe_create_plot(plot_type, cleaned_df, plot_args, color_theme):
    """Create plots with comprehensive error handling"""
    try:
        fig = None
                    
        if plot_type == "Scatter":
            fig = px.scatter(**plot_args)
        elif plot_type == "Line":
            fig = px.line(**plot_args)
        elif plot_type == "Bar":
            fig = px.bar(**plot_args)
        elif plot_type == "Histogram":
            fig = px.histogram(cleaned_df, x=plot_args.get('x'), template=color_theme)
        elif plot_type == "Box":
            fig = px.box(**plot_args)
        elif plot_type == "Violin":
            fig = px.violin(**plot_args)
        elif plot_type == "Heatmap":
            numeric_df = cleaned_df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, text_auto=True, template=color_theme,
                                     title="Correlation Heatmap")
            else:
                return None, "Heatmap requires at least 2 numeric columns"
                        
        if fig:
            fig.update_layout(height=600, title_font_size=20)
            return fig, None
                    
        else:
            return None, f"Could not create {plot_type} plot with the selected parameters"
                    
    except Exception as e:
        return None, f"Error creating {plot_type} plot: {str(e)}"

# --- Main App ---
st.markdown('<h1 class="main-header"> ZynfoGraph </h1>', unsafe_allow_html=True)

# --- Error Display Function ---
def display_error(error_msg, details=None):
    st.markdown(f"""
    <div class="error-container">
        <h4>❌ Error</h4>
        <p>{error_msg}</p>
        {f'<details><summary>Technical Details</summary><pre>{details}</pre></details>' if details else ''}
    </div>
    """, unsafe_allow_html=True)

def display_success(success_msg):
    st.markdown(f"""
    <div class="success-container">
        <h4>✅ Success</h4>
        <p>{success_msg}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
        
    # Initialize session state for uploaded file
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

    # File uploader
    new_uploaded_file = st.file_uploader(
        "📁 Upload your data file",
        type=["csv", "xlsx", "xls", "json", "txt", "tsv", "parquet"],
        help="Supported formats: CSV, Excel, JSON, TXT, TSV, Parquet"
    )

    # Logic to update session state based on new upload or existing file
    if new_uploaded_file is not None:
        # Check if a new file was uploaded or if the existing one changed
        if st.session_state.uploaded_file is None or \
           new_uploaded_file.name != st.session_state.uploaded_file.name or \
           new_uploaded_file.size != st.session_state.uploaded_file.size:
            st.session_state.uploaded_file = new_uploaded_file
            st.success(f"✅ File selected: {st.session_state.uploaded_file.name}")
            st.rerun() # Rerun to process the new file immediately
    elif st.session_state.uploaded_file is not None:
        st.info(f"Using previously uploaded file: {st.session_state.uploaded_file.name}")
    
    # Add a button to clear the uploaded file
    if st.session_state.uploaded_file is not None:
        if st.button("Clear Uploaded File"):
            st.session_state.uploaded_file = None
            st.rerun() # Rerun to clear the file from the UI and re-render the welcome screen

    # Data cleaning options (only show if a file is uploaded)
    cleaning_options = {} # Initialize outside if block
    color_theme = 'plotly_dark' # Default value
    show_advanced = False # Default value

    if st.session_state.uploaded_file:
        st.subheader("🧹 Data Cleaning")
        cleaning_options = {
            'remove_duplicates': st.checkbox("Remove duplicate rows"),
            'handle_missing': st.selectbox(
                "Handle missing values",
                ['keep', 'drop', 'fill_mean', 'fill_median'],
                help="Choose how to handle missing values"
            )
        }
                    
        # Visualization settings
        st.subheader("🎨 Visualization Settings")
        color_theme = st.selectbox(
            "Color Theme",
            ['plotly_dark', 'plotly_white', 'ggplot2', 'seaborn', 'simple_white']
        )
                    
        show_advanced = st.checkbox("Show Advanced Options", value=False)

# --- Main Content ---
# Use st.session_state.uploaded_file for the main logic
if st.session_state.uploaded_file is not None:
    # Load data with error handling
    with st.spinner("Loading data..."):
        try:
            # Pass the file object from session state to load_data
            df, load_message = load_data(st.session_state.uploaded_file)
            if df is not None:
                display_success(load_message)
                            
                # Apply cleaning with error handling
                with st.spinner("Cleaning data..."):
                    cleaned_df, operations, clean_error = safe_clean_data(df, cleaning_options)
                                
                    if cleaned_df is None:
                        display_error("Data cleaning failed", clean_error)
                        st.stop() # Stop execution if cleaning failed
                                
                    if clean_error:
                        st.warning("Data cleaning encountered issues: " + clean_error)
                    elif operations:
                        st.info("🧹 " + " | ".join(operations))
                                
                # Data overview tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Explore", "📈 Visualize", "🧮 Analyze", "📥 Export"])
                            
                with tab1:
                    try:
                        st.subheader("📋 Data Overview")
                                        
                        # Key metrics
                        col1, col2, col3, col4 = st.columns(4)
                                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>📏 Rows</h3>
                                <h2>{cleaned_df.shape[0]:,}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>📊 Columns</h3>
                                <h2>{cleaned_df.shape[1]}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                                        
                        with col3:
                            missing_pct = (cleaned_df.isnull().sum().sum() / (cleaned_df.shape[0] * cleaned_df.shape[1])) * 100
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>❓ Missing</h3>
                                <h2>{missing_pct:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                                        
                        with col4:
                            memory_mb = cleaned_df.memory_usage(deep=True).sum() / 1024 / 1024
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>💾 Memory</h3>
                                <h2>{memory_mb:.1f}MB</h2>
                            </div>
                            """, unsafe_allow_html=True)
                                        
                        st.markdown("---")
                                        
                        # Data preview
                        st.subheader("👀 Data Preview")
                        display_rows = min(100, len(cleaned_df))
                        st.dataframe(cleaned_df.head(display_rows), use_container_width=True, height=400)
                                        
                        # Data types and info
                        col1, col2 = st.columns(2)
                                        
                        with col1:
                            st.subheader("📋 Column Information")
                            info_df = pd.DataFrame({
                                'Column': cleaned_df.columns,
                                'Type': cleaned_df.dtypes.astype(str),
                                'Non-Null': cleaned_df.count(),
                                'Missing': cleaned_df.isnull().sum(),
                                'Missing %': (cleaned_df.isnull().sum() / len(cleaned_df) * 100).round(2)
                            })
                            st.dataframe(info_df, use_container_width=True)
                                        
                        with col2:
                            st.subheader("📊 Statistical Summary")
                            try:
                                st.dataframe(cleaned_df.describe(), use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not generate statistical summary: {str(e)}")
                                        
                    except Exception as e:
                        display_error("Error in Overview tab", str(e))
                            
                with tab2:
                    try:
                        st.subheader("🔍 Data Exploration")
                                        
                        # Column selection for exploration
                        explore_col = st.selectbox("Select column to explore", cleaned_df.columns)
                                        
                        if explore_col:
                            col1, col2 = st.columns(2)
                                            
                            with col1:
                                st.write(f"**Column: {explore_col}**")
                                st.write(f"Data Type: {cleaned_df[explore_col].dtype}")
                                st.write(f"Unique Values: {cleaned_df[explore_col].nunique()}")
                                st.write(f"Missing Values: {cleaned_df[explore_col].isnull().sum()}")
                                                
                                try:
                                    if cleaned_df[explore_col].dtype in ['object']:
                                        st.write("**Top Values:**")
                                        st.write(cleaned_df[explore_col].value_counts().head(10))
                                    else:
                                        st.write("**Statistics:**")
                                        st.write(cleaned_df[explore_col].describe())
                                except Exception as e:
                                    st.warning(f"Could not generate column statistics: {str(e)}")
                                                
                            with col2:
                                try:
                                    # Auto-generate appropriate visualization
                                    if cleaned_df[explore_col].dtype in ['object']:
                                        # Categorical data - bar chart
                                        value_counts = cleaned_df[explore_col].value_counts().head(20)
                                        fig = px.bar(
                                            x=value_counts.values,
                                            y=value_counts.index,
                                            orientation='h',
                                            title=f"Distribution of {explore_col}",
                                            template=color_theme
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        # Numeric data - histogram
                                        fig = px.histogram(
                                            cleaned_df,
                                            x=explore_col,
                                            title=f"Distribution of {explore_col}",
                                            template=color_theme
                                        )
                                        fig.update_layout(height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not create visualization for {explore_col}: {str(e)}")
                                                
                        # Data filtering
                        st.subheader("🔧 Data Filtering")
                                        
                        try:
                            # Numeric filters
                            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
                            if len(numeric_cols) > 0:
                                filter_col = st.selectbox("Select numeric column to filter", ["None"] + list(numeric_cols))
                                if filter_col != "None":
                                    min_val, max_val = float(cleaned_df[filter_col].min()), float(cleaned_df[filter_col].max())
                                    if min_val != max_val: # Avoid slider error when min == max
                                        filter_range = st.slider(
                                            f"Filter {filter_col}",
                                            min_val, max_val, (min_val, max_val)
                                        )
                                        filtered_df = cleaned_df[
                                            (cleaned_df[filter_col] >= filter_range[0]) & 
                                            (cleaned_df[filter_col] <= filter_range[1])
                                        ]
                                        st.write(f"Filtered data: {len(filtered_df)} rows")
                                        st.dataframe(filtered_df.head(), use_container_width=True)
                                    else:
                                        st.info(f"Column {filter_col} has constant values, cannot filter")
                                else:
                                    st.info("No numeric columns available for filtering")
                        except Exception as e:
                            display_error("Error in data filtering", str(e))
                                        
                    except Exception as e:
                        display_error("Error in Explore tab", str(e))
                            
                with tab3:
                    try:
                        st.subheader("📈 Advanced Visualizations")
                                        
                        # Enhanced plot options
                        col1, col2, col3 = st.columns(3)
                                        
                        with col1:
                            plot_type = st.selectbox(
                                "📊 Plot Type",
                                ["Scatter", "Line", "Bar", "Histogram", "Box", "Violin", "Heatmap"]
                            )
                                        
                        with col2:
                            x_col = st.selectbox("📌 X-axis", ["None"] + list(cleaned_df.columns))
                                        
                        with col3:
                            y_col = st.selectbox("📌 Y-axis", ["None"] + list(cleaned_df.columns))
                                        
                        # Additional options
                        if show_advanced:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                color_col = st.selectbox("🎨 Color by", ["None"] + list(cleaned_df.columns))
                            with col2:
                                size_col = st.selectbox("📏 Size by", ["None"] + list(cleaned_df.select_dtypes(include=[np.number]).columns))
                            with col3:
                                facet_col = st.selectbox("🔄 Facet by", ["None"] + list(cleaned_df.columns))
                        else:
                            color_col = size_col = facet_col = "None"
                                        
                        # Generate plots
                        if plot_type and x_col != "None":
                            # Prepare arguments
                            plot_args = {
                                'data_frame': cleaned_df,
                                'template': color_theme
                            }
                                            
                            if x_col != "None":
                                plot_args['x'] = x_col
                            if y_col != "None":
                                plot_args['y'] = y_col
                            if color_col != "None":
                                plot_args['color'] = color_col
                            if size_col != "None":
                                plot_args['size'] = size_col
                            if facet_col != "None":
                                plot_args['facet_col'] = facet_col
                                            
                            # Create plot with error handling
                            fig, plot_error = safe_create_plot(plot_type, cleaned_df, plot_args, color_theme)
                                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                                
                                # Enhanced download options
                                st.subheader("📥 Download Plot")
                                col1, col2, col3 = st.columns(3)
                                                
                                with col1:
                                    if st.button("📸 Download as PNG"):
                                        with st.spinner("Generating PNG..."):
                                            img_data, filename, mime_type, export_error = export_plot_as_image(
                                                fig, "png", f"{plot_type.lower()}_plot"
                                            )
                                            if img_data:
                                                st.download_button(
                                                    label="📥 Download PNG",
                                                    data=img_data,
                                                    file_name=filename,
                                                    mime=mime_type
                                                )
                                                if export_error:
                                                    st.warning(export_error)
                                            else:
                                                display_error("PNG export failed", export_error)
                                                
                                with col2:
                                    if st.button("📄 Download as PDF"):
                                        with st.spinner("Generating PDF..."):
                                            img_data, filename, mime_type, export_error = export_plot_as_image(
                                                fig, "pdf", f"{plot_type.lower()}_plot"
                                            )
                                            if img_data:
                                                st.download_button(
                                                    label="📥 Download PDF",
                                                    data=img_data,
                                                    file_name=filename,
                                                    mime=mime_type
                                                )
                                                if export_error:
                                                    st.warning(export_error)
                                            else:
                                                display_error("PDF export failed", export_error)
                                                
                                with col3:
                                    if st.button("🌐 Download as HTML"):
                                        html_str = fig.to_html()
                                        st.download_button(
                                            label="📥 Download HTML",
                                            data=html_str,
                                            file_name=f"{plot_type.lower()}_plot.html",
                                            mime="text/html"
                                        )
                            else:
                                display_error("Plot creation failed", plot_error)
                                        
                    except Exception as e:
                        display_error("Error in Visualize tab", str(e))
                            
                with tab4:
                    try:
                        st.subheader("🧮 Statistical Analysis")
                                        
                        # Correlation analysis
                        numeric_df = cleaned_df.select_dtypes(include=[np.number])
                        if len(numeric_df.columns) > 1:
                            st.subheader("🔗 Correlation Analysis")
                                            
                            try:
                                # Correlation matrix
                                corr_matrix = numeric_df.corr()
                                                
                                col1, col2 = st.columns(2)
                                                
                                with col1:
                                    fig = px.imshow(
                                        corr_matrix,
                                        text_auto=True,
                                        aspect="auto",
                                        template=color_theme,
                                        title="Correlation Matrix"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                                
                                with col2:
                                    st.write("**Strongest Correlations:**")
                                    # Get correlation pairs
                                    corr_pairs = []
                                    for i in range(len(corr_matrix.columns)):
                                        for j in range(i+1, len(corr_matrix.columns)):
                                            corr_pairs.append({
                                                'Variable 1': corr_matrix.columns[i],
                                                'Variable 2': corr_matrix.columns[j],
                                                'Correlation': corr_matrix.iloc[i, j]
                                            })
                                                
                                    if corr_pairs:
                                        corr_df = pd.DataFrame(corr_pairs)
                                        corr_df = corr_df.reindex(corr_df['Correlation'].abs().sort_values(ascending=False).index)
                                        st.dataframe(corr_df.head(10), use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not generate correlation analysis: {str(e)}")
                                                
                        # Outlier detection
                        st.subheader("🎯 Outlier Detection")
                        if len(numeric_df.columns) > 0:
                            outlier_col = st.selectbox("Select column for outlier detection", numeric_df.columns)
                                            
                            if outlier_col:
                                try:
                                    # Calculate outliers using IQR method
                                    Q1 = cleaned_df[outlier_col].quantile(0.25)
                                    Q3 = cleaned_df[outlier_col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - 1.5 * IQR
                                    upper_bound = Q3 + 1.5 * IQR
                                                
                                    outliers = cleaned_df[
                                        (cleaned_df[outlier_col] < lower_bound) | 
                                        (cleaned_df[outlier_col] > upper_bound)
                                    ]
                                                
                                    col1, col2 = st.columns(2)
                                                
                                    with col1:
                                        st.metric("Outliers Found", len(outliers))
                                        st.metric("Outlier Percentage", f"{len(outliers)/len(cleaned_df)*100:.2f}%")
                                                
                                    with col2:
                                        # Box plot for outlier visualization
                                        fig = px.box(cleaned_df, y=outlier_col, template=color_theme,
                                                               title=f"Box Plot - {outlier_col}")
                                        st.plotly_chart(fig, use_container_width=True)
                                                
                                    if len(outliers) > 0:
                                        st.write("**Outlier Records:**")
                                        st.dataframe(outliers, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not perform outlier detection: {str(e)}")
                        else:
                            st.info("No numeric columns available for outlier detection")
                                        
                    except Exception as e:
                        display_error("Error in Analyze tab", str(e))
                            
                with tab5:
                    try:
                        st.subheader("📥 Export & Download")
                                        
                        col1, col2 = st.columns(2)
                                        
                        with col1:
                            st.write("**Export Processed Data**")
                                            
                            try:
                                # CSV download
                                csv = cleaned_df.to_csv(index=False)
                                st.download_button(
                                    label="📄 Download as CSV",
                                    data=csv,
                                    file_name=f"processed_{st.session_state.uploaded_file.name.split('.')[0]}.csv",
                                    mime="text/csv"
                                )
                                                
                                # Excel download
                                buffer = BytesIO()
                                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                    cleaned_df.to_excel(writer, sheet_name='Data', index=False)
                                    if len(numeric_df.columns) > 1:
                                        try:
                                            corr_matrix = numeric_df.corr()
                                            corr_matrix.to_excel(writer, sheet_name='Correlations')
                                        except:
                                            pass # Skip correlations if error
                                                
                                st.download_button(
                                    label="📊 Download as Excel",
                                    data=buffer.getvalue(),
                                    file_name=f"processed_{st.session_state.uploaded_file.name.split('.')[0]}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            except Exception as e:
                                display_error("Error preparing data exports", str(e))
                                                
                        with col2:
                            st.write("**Data Summary Report**")
                                            
                            try:
                                # Generate summary report
                                profile, profile_error = safe_get_data_profile(cleaned_df)
                                                
                                if profile:
                                    report = f"""# Data Analysis Report
**File:** {st.session_state.uploaded_file.name}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Overview
- **Rows:** {profile['shape'][0]:,}
- **Columns:** {profile['shape'][1]}
- **Memory Usage:** {profile['memory_usage']/1024/1024:.2f} MB
- **Duplicate Rows:** {profile['duplicate_rows']}

## Column Types
- **Numeric:** {len(profile['numeric_columns'])}
- **Categorical:** {len(profile['categorical_columns'])}
- **DateTime:** {len(profile['datetime_columns'])}

## Data Quality
- **Missing Values:** {profile['missing_values'].sum()}
- **Completeness:** {((cleaned_df.shape[0] * cleaned_df.shape[1] - profile['missing_values'].sum()) / (cleaned_df.shape[0] * cleaned_df.shape[1]) * 100):.2f}%

## Applied Operations
{chr(10).join([f"- {op}" for op in operations]) if operations else "- No cleaning operations applied"}"""
                                                
                                    st.download_button(
                                        label="📋 Download Report",
                                        data=report,
                                        file_name=f"report_{st.session_state.uploaded_file.name.split('.')[0]}.md",
                                        mime="text/markdown"
                                    )
                                else:
                                    display_error("Could not generate report", profile_error)
                            except Exception as e:
                                display_error("Error generating report", str(e))
                                                
                    except Exception as e:
                        display_error("Error in Export tab", str(e))
            else:
                display_error("Failed to load data", load_message)
        except MemoryError as me:
            display_error("Memory Error: The dataset is too large for the application to process.", 
                          f"Please try a smaller file or consider running this application on a machine with more RAM. Details: {str(me)}")
            st.session_state.uploaded_file = None # Clear file to allow user to try again
            st.rerun() # Rerun to show the welcome screen after error
        except Exception as e:
            display_error("An unexpected error occurred during data loading or processing.", 
                          f"Please try again or upload a different file. Details: {str(e)}\n{traceback.format_exc()}")
            st.session_state.uploaded_file = None # Clear file to allow user to try again
            st.rerun() # Rerun to show the welcome screen after error
else:
    # Welcome screen - Fixed HTML rendering
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>🚀Transform, analyze, and visualize your data with powerful tools and insights!</h2>
        <p style="font-size: 1.2rem; margin: 2rem 0;">
            Upload your data file to get started with powerful analysis and visualization tools.
        </p>
    </div>
    """, unsafe_allow_html=True)
        
    # Use Streamlit columns instead of HTML grid for better compatibility
    st.markdown("### ✨ Key Features")
        
    col1, col2, col3, col4 = st.columns(4)
        
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Smart Visualizations</h3>
            <p>Auto-generated charts with PNG/PDF export</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🧹 Data Cleaning</h3>
            <p>Handle missing values, duplicates, and outliers</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Deep Analysis</h3>
            <p>Statistical insights and correlation analysis</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📥 Easy Export</h3>
            <p>Download processed data and comprehensive reports</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
        
    # Info section
    col1, col2 = st.columns(2)
        
    with col1:
        st.info("**Supported formats:** CSV, Excel, JSON, TXT, TSV, Parquet")
    with col2:
        st.success("**New features:** PNG/PDF plot exports, Enhanced error handling, Better file format detection")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <style>
    /* Floating effect */
    @keyframes float-icons {
        0%, 100% {
            transform: translateY(0);
        }
        50% {
            transform: translateY(-8px);
        }
    }
    /* Styling for the GitHub and Portfolio icons */
    .social-icon {
        animation: float-icons 3s ease-in-out infinite; /* Floating animation */
        transition: transform 0.2s ease, filter 0.2s ease;
        filter: invert(100%);
        margin: 0 12px;
    }
    /* Hover effect with glow */
    .social-icon:hover {
        transform: translateY(-10px) scale(1.1); /* Slightly bigger + lifted */
        filter: invert(100%) drop-shadow(0 0 10px #ffffff) brightness(1.3); /* Glow */
        cursor: pointer;
    }
    </style>
    <center>
    <small>
    "Predicting the future isn’t magic, it’s artificial intelligence."<br>
     Developed by <strong>Nandeesh</strong><br><br>
    <a href="https://github.com/nandeesh71" target="_blank" style="text-decoration: none;">
        <img class="social-icon" src="https://raw.githubusercontent.com/simple-icons/simple-icons/develop/icons/github.svg" alt="GitHub" width="32" />
    </a>
    <a href="https://nandeesh-71.web.app/" target="_blank" style="text-decoration: none;">
        <img class="social-icon" src="https://cdn-icons-png.flaticon.com/512/1159/1159607.png" alt="Portfolio" width="32" />
    </a><br><br>
    © 2025 Team Elites. All rights reserved.
    </small>
    </center>
    """,
    unsafe_allow_html=True)
