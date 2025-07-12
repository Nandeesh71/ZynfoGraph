# 📊 ZynfoGraph

**ZynfoGraph** is an advanced, visually stunning, and user-friendly Streamlit application that empowers users to upload, clean, analyze, and visualize datasets effortlessly. Whether you're a data enthusiast, analyst, or student, ZynfoGraph provides interactive dashboards, smart visualizations, and export-ready insights — all from your browser.

![ZynfoGraph](https://i.postimg.cc/nrL8ScR2/Zynfo-Graph-icon.jpg)

---

## 🚀 Features

- 📁 **Multi-format Support**: Upload CSV, Excel, JSON, TXT, TSV, and Parquet files.
- 🧹 **Smart Data Cleaning**: Remove duplicates, handle missing values, and prepare clean datasets.
- 📈 **Advanced Visualizations**: Generate scatter, line, bar, histogram, violin, box, and heatmap plots with Plotly.
- 📊 **Deep Data Profiling**: View memory usage, missing values, column types, and summary stats.
- 📥 **Download Options**: Export plots as PNG/PDF/HTML and download cleaned data as CSV or Excel.
- 📋 **Insightful Reports**: Generate markdown reports detailing your dataset.

---

## 🖼️ Screenshots

| Dashboard Overview | Interactive Visuals |
|-------------------|---------------------|
| ![Overview](https://i.postimg.cc/0N5Dwsmz/overview-sample.png) | ![Visuals](https://i.postimg.cc/VLhjWDvw/plot-sample.png) |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/nandeesh71/ZynfoGraph.git
cd ZynfoGraph

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run ZynfoGraph.py
