# MetaVision

**Biomolecule Data Processing Dashboard & 3D Visualization Platform**

MetaVision is an advanced web-based platform designed for processing, analyzing, and visualizing 3D biomolecular data, specifically tailored for MALDI-MS imaging workflows. It provides a comprehensive pipeline from raw data upload to high-fidelity 3D visualization and export.

## 🚀 Key Features

### 1. Data Processing Pipeline

* **Data Upload**: Support for large CSV files with chunked upload capabilities (up to 2GB).
* **Normalization (MetaNorm3D)**: Implements Total Sum/Ion Count (TIC) and section-specific normalization standardization.
* **3D Alignment (MetaAlign3D)**: Motion correction and spatial alignment of tissue sections using the ECC (Enhanced Correlation Coefficient) algorithm.
* **Processing**:
  * **MetaInterp3D**: Spatial interpolation to handle missing data points.
  * **MetaImpute3D**: Neighbor-based imputation for robust data enhancement.

### 2. Visualization & Analysis

* **Interactive 3D Visualization**: Real-time rendering of processed biomolecular data.
* **Slice Navigation**: Explore data through 2D slice views with mouse wheel navigation.
* **Dashboard**: A user-friendly interface to manage the entire workflow.

### 3. Export

* **MetaAtlas3D**: Export processed data to NIfTI format for compatibility with external medical imaging tools.

## 🛠️ Tech Stack

* **Backend/Frontend Framework**: [Dash](https://dash.plotly.com/) (Python)
* **Image Processing**: OpenCV, Scikit-Image, SciPy
* **Medical Imaging Formats**: Nibabel
* **Data Manipulation**: Pandas, NumPy
* **Caching**: Flask-Caching
* **Dependency Management**: `uv` or `pip`

## 📦 Installation

### Prerequisites

* Python **3.13** or higher

### Steps

1. **Clone the repository**

    ```bash
    git clone https://github.com/vamsi-arumalla/metavision.git
    cd metavision
    ```

2. **Install Dependencies**

    Using **uv** (Recommended):

    ```bash
    uv sync
    ```

    OR using **pip**:

    ```bash
    pip install .
    ```

## 🖥️ Usage

1. **Start the Application**

    ```bash
    python main.py
    ```

2. **Access the Dashboard**
    Open your web browser and navigate to:
    `http://127.0.0.1:8050/`

3. **Workflow**
    * Click **"Let's Go!"** on the landing page.
    * **Upload** your CSV data files.
    * Follow the pipeline steps: **Upload -> Normalize -> Align -> Process**.
    * Use the **Visualization** tab to explore the 3D results.
    * Use the **Export** tab to save your data.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

[Add License Information Here]
