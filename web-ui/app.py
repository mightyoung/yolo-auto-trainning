"""
YOLO Auto-Training System - Streamlit Web UI
Location: web-ui/app.py

A modern, clean web interface for the YOLO Auto-Training system.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# ==================== Configuration ====================

st.set_page_config(
    page_title="YOLO Auto-Training",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #06d6a0;
        --secondary: #00b4d8;
        --background: #0a0a0b;
        --surface: #111113;
        --text: #fafafa;
        --text-muted: #a1a1aa;
    }

    /* Override Streamlit colors */
    .stApp {
        background: var(--background);
        color: var(--text);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: var(--surface);
    }

    /* Cards */
    div.stButton > button {
        background: linear-gradient(135deg, #06d6a0, #00b4d8);
        border: none;
        color: #0a0a0b;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        transition: all 0.2s;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(6, 214, 160, 0.3);
    }

    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }

    /* DataFrames */
    div[data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #06d6a0, #00b4d8);
    }
</style>
""", unsafe_allow_html=True)

# ==================== API Configuration ====================

# Configure your Business API URL here
API_BASE = "http://localhost:8000"


def api_get(endpoint):
    """Make GET request to API."""
    try:
        response = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        return None


def api_post(endpoint, data, timeout=30):
    """Make POST request to API."""
    try:
        response = requests.post(f"{API_BASE}{endpoint}", json=data, timeout=timeout)
        if response.status_code in (200, 201):
            return response.json()
        return None
    except Exception as e:
        return None


# Initialize session state for navigation
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"


# ==================== Page Functions ====================

def page_dashboard():
    """Dashboard page showing system overview."""
    st.title("🚀 Dashboard")
    st.markdown("### System Overview")

    # Get health info
    health = api_get("/health")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if health and health.get("training_api") == "available":
            st.metric("Training API", "Connected", delta="✓")
        else:
            st.metric("Training API", "Disconnected", delta="✗")

    with col2:
        if health and health.get("redis") == "connected":
            st.metric("Database", "Connected", delta="✓")
        else:
            st.metric("Database", "Disconnected", delta="✗")

    with col3:
        st.metric("Datasets", "—", delta="?")

    with col4:
        st.metric("Models", "—", delta="?")

    st.divider()

    # Recent jobs
    st.markdown("### Recent Training Jobs")

    # Get training jobs from Redis or storage
    # For now, show a message if no jobs
    st.info("Training jobs will appear here after submission.")

    st.markdown("""
    **To start a training job:**
    1. Go to **Data Discovery** to find and download a dataset
    2. Go to **Training** to configure and start training
    3. View progress here on the Dashboard
    """)

    # Quick actions
    st.markdown("### Quick Actions")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        if st.button("🎯 Start Training", use_container_width=True):
            st.session_state.page = "Training"
            st.rerun()

    with c2:
        if st.button("🔍 Discover Data", use_container_width=True):
            st.session_state.page = "Data Discovery"
            st.rerun()

    with c3:
        if st.button("🏷️ Auto Label", use_container_width=True):
            st.session_state.page = "Auto Label"
            st.rerun()

    with c4:
        if st.button("📊 Analyze Data", use_container_width=True):
            st.session_state.page = "Data Analysis"
            st.rerun()

    with c5:
        if st.button("📦 Export Model", use_container_width=True):
            st.session_state.page = "Models"
            st.rerun()


def page_data_discovery():
    """Data discovery page."""
    st.title("🔍 Data Discovery")
    st.markdown("### Search Datasets")

    # Search bar
    query = st.text_input("Search for datasets", placeholder="e.g., car detection, traffic sign...")

    col1, col2 = st.columns([3, 1])

    with col1:
        source_filter = st.multiselect(
            "Data Sources",
            ["Roboflow", "Kaggle", "HuggingFace"],
            default=["Roboflow", "Kaggle", "HuggingFace"]
        )

    with col2:
        min_images = st.number_input("Min Images", min_value=100, value=1000, step=100)

    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching datasets..."):
            # Call real API
            result = api_post("/api/v1/data/search", {
                "query": query,
                "max_results": 10,
                "sources": source_filter,
                "min_images": min_images
            })

            if result and result.get("datasets"):
                st.success(f"Found {len(result['datasets'])} datasets!")

                for r in result["datasets"]:
                    with st.expander(f"{r['name']} ({r['source']})"):
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.metric("Images", f"{r.get('images', 'N/A'):,}")
                        with c2:
                            st.metric("License", r.get('license', 'Unknown'))
                        with c3:
                            st.metric("Match Score", f"{r.get('relevance_score', 0)*100:.0f}%")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.button(f"📥 Download {r['name']}", key=f"dl_{r['name']}")
                        with col2:
                            if r.get('url'):
                                st.markdown(f"[🔗 View Source]({r['url']})")
            else:
                st.warning("No datasets found. Try a different search query.")


def page_training():
    """Training configuration page."""
    st.title("🎯 Training")
    st.markdown("### Configure Training")

    # Check API connection
    health = api_get("/health")
    if not health or health.get("training_api") != "available":
        st.warning("⚠️ Training API is not connected. Please ensure the GPU server is running.")

    # Model selection
    st.markdown("#### Model Selection")

    models = {
        "yolo11n": {"name": "YOLO11n", "params": "2.6M", "fps": "100+", "gpu": "2GB"},
        "yolo11s": {"name": "YOLO11s", "params": "9.7M", "fps": "60+", "gpu": "4GB"},
        "yolo11m": {"name": "YOLO11m", "params": "25.9M", "fps": "40+", "gpu": "8GB"},
        "yolo11l": {"name": "YOLO11l", "params": "51.5M", "fps": "25+", "gpu": "12GB"},
        "yolo11x": {"name": "YOLO11x", "params": "97.2M", "fps": "15+", "gpu": "16GB"},
    }

    cols = st.columns(5)

    selected_model = None
    for idx, (model_key, info) in enumerate(models.items()):
        with cols[idx]:
            if st.button(
                f"**{info['name']}**\n\n{info['params']}\n{info['fps']} FPS",
                use_container_width=True,
                key=f"model_{model_key}"
            ):
                selected_model = model_key

    if selected_model:
        st.success(f"Selected: {models[selected_model]['name']}")

    st.divider()

    # Configuration
    st.markdown("#### Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=100)

    with col2:
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)

    with col3:
        image_size = st.selectbox("Image Size", [320, 416, 512, 640, 1280], index=3)

    # Dataset path input
    st.markdown("#### Dataset")
    data_yaml = st.text_input(
        "Dataset YAML Path",
        placeholder="e.g., datasets/coco8/data.yaml",
        help="Path to your dataset configuration file"
    )

    # Resource estimation
    st.divider()
    st.markdown("#### Estimated Resources")

    if selected_model:
        c1, c2 = st.columns(2)

        with c1:
            st.metric("GPU Memory", models[selected_model]["gpu"])

        with c2:
            est_time = epochs * 2
            st.metric("Est. Training Time", f"~{est_time} min")

    # Start training
    st.divider()

    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        if not selected_model:
            st.error("Please select a model first.")
        elif not data_yaml:
            st.error("Please enter a dataset YAML path.")
        else:
            with st.spinner("Submitting training job..."):
                result = api_post("/api/v1/train/submit", {
                    "model": selected_model,
                    "data_yaml": data_yaml,
                    "epochs": epochs,
                    "imgsz": image_size
                })

                if result:
                    st.success(f"Training job submitted! Task ID: {result.get('task_id')}")
                    st.info(f"Estimated time: ~{result.get('estimated_time_minutes', 'N/A')} minutes")
                    st.info("Check the Dashboard for progress updates.")
                else:
                    st.error("Failed to submit training job. Please check API connection.")


def page_models():
    """Models management page."""
    st.title("📦 Models")
    st.markdown("### Trained Models")

    # Info message
    st.info("Trained models will appear here after training completion.")

    # Placeholder for model list - would come from storage/database
    st.markdown("""
    **After training completes, your models will be listed here.**

    You can then:
    - Export models to different formats (ONNX, TensorRT)
    - Download models for deployment
    - View performance metrics
    """)

    # Model export section
    st.divider()
    st.markdown("### Export Model")

    col1, col2 = st.columns(2)

    with col1:
        export_model_path = st.text_input(
            "Model Path",
            placeholder="e.g., runs/train/exp/weights/best.pt"
        )

    with col2:
        export_platform = st.selectbox(
            "Target Platform",
            ["jetson_orin", "jetson_nano", "jetson_xavier", "rk3588", "onnx", "tensorrt"]
        )

    if st.button("📤 Export Model", type="primary", use_container_width=True):
        if not export_model_path:
            st.error("Please enter a model path.")
        else:
            with st.spinner("Submitting export job..."):
                result = api_post("/api/v1/deploy/export", {
                    "model_path": export_model_path,
                    "platform": export_platform,
                    "imgsz": 640
                })

                if result:
                    st.success(f"Export job submitted! Task ID: {result.get('task_id')}")
                else:
                    st.error("Failed to submit export job.")


def page_data_analysis():
    """Data Analysis page using DeepAnalyze."""
    st.title("📊 Data Analysis")
    st.markdown("### AI-Powered Data Analysis with DeepAnalyze")

    # Check DeepAnalyze API availability
    analysis_health = api_get("/api/v1/analysis/health")

    if not analysis_health or analysis_health.get("status") != "available":
        st.warning("⚠️ DeepAnalyze API is not available. Please ensure the service is running.")
        st.markdown("""
        **To enable Data Analysis:**
        1. Deploy DeepAnalyze-8B using vLLM
        2. Start the DeepAnalyze API server
        3. Configure DEEPANALYZE_API_URL in environment
        """)
        return

    # Analysis options
    st.markdown("#### Analysis Type")

    analysis_type = st.selectbox(
        "Select Analysis",
        ["quality", "distribution", "anomalies", "full"],
        format_func=lambda x: {
            "quality": "Data Quality Analysis",
            "distribution": "Data Distribution Analysis",
            "anomalies": "Anomaly Detection",
            "full": "Comprehensive Analysis"
        }[x]
    )

    # Dataset path input
    st.markdown("#### Dataset")

    dataset_path = st.text_input(
        "Dataset Path",
        placeholder="e.g., datasets/my_data.csv or datasets/",
        help="Path to a data file (CSV, Excel, JSON) or directory containing data files"
    )

    # Custom prompt option
    use_custom_prompt = st.checkbox("Use custom analysis prompt")
    custom_prompt = ""

    if use_custom_prompt:
        custom_prompt = st.text_area(
            "Custom Prompt",
            placeholder="Enter your analysis requirements...",
            height=100
        )

    # Analyze button
    if st.button("🔍 Analyze Data", type="primary", use_container_width=True):
        if not dataset_path:
            st.error("Please enter a dataset path.")
        else:
            with st.spinner("Analyzing data with DeepAnalyze... This may take a while."):
                request_data = {
                    "dataset_path": dataset_path,
                    "analysis_type": analysis_type
                }

                if use_custom_prompt and custom_prompt:
                    request_data["prompt"] = custom_prompt

                result = api_post("/api/v1/analysis/analyze", request_data, timeout=300)

                if result:
                    if result.get("status") == "completed":
                        st.success("Analysis completed!")

                        # Display analysis content
                        if result.get("content"):
                            st.markdown("### Analysis Results")
                            st.markdown(result["content"])

                        # Display generated files
                        if result.get("files"):
                            st.markdown("### Generated Files")
                            for f in result["files"]:
                                st.markdown(f"- [{f['name']}]({f['url']})")
                    else:
                        st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
                else:
                    st.error("Failed to submit analysis request.")

    # Info section
    st.divider()
    st.markdown("""
    ### About DeepAnalyze

    DeepAnalyze is an autonomous data science AI agent that can:
    - Analyze data quality (missing values, outliers, duplicates)
    - Explore data distributions and correlations
    - Detect anomalies and unusual patterns
    - Generate comprehensive data science reports
    """)


def page_auto_label():
    """Auto Labeling page using AutoDistill."""
    st.title("🏷️ Auto Labeling")
    st.markdown("### AI-Powered Automatic Image Labeling")

    # Check API availability
    health = api_get("/health")

    if not health or health.get("training_api") != "available":
        st.warning("⚠️ Training API is not connected. Please ensure the GPU server is running.")
        return

    # Base model selection
    st.markdown("#### Base Model")

    base_model = st.selectbox(
        "Select Base Model",
        ["grounded_sam", "grounding_dino", "owlv2"],
        format_func=lambda x: {
            "grounded_sam": "GroundedSAM (Recommended - Best accuracy)",
            "grounding_dino": "GroundingDINO (Fast, Open-set)",
            "owlv2": "OWLv2 (Zero-shot)"
        }[x]
    )

    # Confidence threshold
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        help="Lower values = more detections but potentially more false positives"
    )

    # Class input
    st.markdown("#### Classes to Detect")

    classes_input = st.text_area(
        "Enter classes (one per line)",
        placeholder="person\ncar\ndog\ncat",
        help="Enter each class name on a new line"
    )

    # Input folder
    st.markdown("#### Input Data")

    input_folder = st.text_input(
        "Image Folder Path",
        placeholder="e.g., datasets/my_images",
        help="Path to folder containing images to label"
    )

    # Output folder
    output_folder = st.text_input(
        "Output Folder Path (Optional)",
        placeholder="e.g., datasets/labeled",
        help="Where to save labeled dataset"
    )

    # Submit button
    if st.button("🏷️ Start Auto Labeling", type="primary", use_container_width=True):
        if not classes_input:
            st.error("Please enter at least one class to detect.")
        elif not input_folder:
            st.error("Please enter an input folder path.")
        else:
            classes = [c.strip() for c in classes_input.split("\n") if c.strip()]

            with st.spinner("Submitting labeling job..."):
                result = api_post("/api/v1/label/submit", {
                    "task_id": f"label_{int(time.time())}",
                    "input_folder": input_folder,
                    "classes": classes,
                    "base_model": base_model,
                    "conf_threshold": conf_threshold
                }, timeout=30)

                if result:
                    st.success(f"Labeling job submitted! Task ID: {result.get('task_id')}")
                    st.info(f"Base Model: {base_model}")
                    st.info(f"Classes: {', '.join(classes)}")

                    # Show status check info
                    st.markdown("""
                    **Next Steps:**
                    1. Check job status in Dashboard
                    2. Once complete, the labeled dataset will be saved to the output folder
                    3. Use the labeled dataset to train a model
                    """)
                else:
                    st.error("Failed to submit labeling job.")

    # Info section
    st.divider()
    st.markdown("""
    ### About Auto Labeling

    Auto labeling uses foundation models to automatically annotate your images:

    | Model | Accuracy | Speed | Best For |
    |-------|----------|-------|----------|
    | GroundedSAM | Highest | Slow | General object detection |
    | GroundingDINO | High | Fast | Open-set detection |
    | OWLv2 | Medium | Fast | Quick prototyping |

    **Output Format:** YOLO format (images/ + annotations/ + data.yaml)
    """)


# ==================== Main App ====================

def main():
    """Main application."""

    # Sidebar navigation
    with st.sidebar:
        st.title("🚀 YOLO Auto-Train")

        st.markdown("---")

        # Use session state for navigation
        nav_options = ["Dashboard", "Data Discovery", "Auto Label", "Data Analysis", "Training", "Models"]
        if st.session_state.page not in nav_options:
            st.session_state.page = "Dashboard"

        page = st.radio(
            "Navigation",
            nav_options,
            index=nav_options.index(st.session_state.page)
        )

        # Update session state when sidebar changes
        if page != st.session_state.page:
            st.session_state.page = page

        st.markdown("---")

        # System status - check API connectivity
        st.markdown("### System Status")

        # Check API status
        health = api_get("/health") if api_get else None

        c1, c2 = st.columns(2)

        with c1:
            if health:
                st.metric("API", "Online", delta="✓")
            else:
                st.metric("API", "Offline", delta="✗")

        with c2:
            st.metric("GPU", "Available", delta="?")

        st.markdown("---")

        # Settings
        st.markdown("### Settings")

        api_url = st.text_input("API URL", value=API_BASE)
        if api_url != API_BASE:
            st.rerun()

    # Route to selected page
    if st.session_state.page == "Dashboard":
        page_dashboard()
    elif st.session_state.page == "Data Discovery":
        page_data_discovery()
    elif st.session_state.page == "Auto Label":
        page_auto_label()
    elif st.session_state.page == "Data Analysis":
        page_data_analysis()
    elif st.session_state.page == "Training":
        page_training()
    elif st.session_state.page == "Models":
        page_models()


if __name__ == "__main__":
    main()
