# 🧭 Advanced Pathfinder Dashboard Setup Guide

## 📊 **Interactive Streamlit Dashboard**

The `pathfinder/dashboard.py` provides a comprehensive interactive dashboard with:
- **Dynamic weight adjustment** for 5 career dimensions
- **Real-time ranking** updates based on your priorities  
- **Radar plots** for multi-dimensional comparison
- **Interactive heatmaps** and salary analysis
- **Professional visualizations** using Plotly

---

## 🛠️ **Installation Options**

### **Option 1: Virtual Environment (Recommended)**
```bash
# Create and activate virtual environment
python3 -m venv streamlit_env
source streamlit_env/bin/activate  # On Windows: streamlit_env\Scripts\activate

# Install required packages
pip install streamlit plotly pandas numpy

# Run the dashboard
streamlit run pathfinder/dashboard.py
```

### **Option 2: System Installation (Not Recommended)**
```bash
# Install system-wide (may cause conflicts)
pip3 install streamlit plotly pandas numpy

# Run the dashboard
streamlit run pathfinder/dashboard.py
```

### **Option 3: Conda Environment**
```bash
# Create conda environment
conda create -n pathfinder python=3.9
conda activate pathfinder

# Install packages
conda install streamlit plotly pandas numpy

# Run the dashboard
streamlit run pathfinder/dashboard.py
```

---

## 🚀 **Quick Start Instructions**

### **1. Check Prerequisites**
```bash
# Verify you have the metrics data
ls pathfinder/output/full_metrics_raw.csv

# If missing, generate it first:
python3 pathfinder/metrics_engine.py
```

### **2. Install Dependencies**
Choose one installation option above, then verify:
```bash
python3 -c "import streamlit, plotly, pandas; print('✅ All dependencies available')"
```

### **3. Launch Dashboard**
```bash
streamlit run pathfinder/dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## 📊 **Dashboard Features**

### 🎛️ **Interactive Controls**
- **Weight Sliders**: Adjust importance of each dimension
  - 🧑‍💻 Job Market (20% default)
  - 💵 Compensation (30% default)  
  - ⚔️ Accessibility (20% default)
  - 🔧 Skill Compatibility (20% default)
  - 🚀 Future Forecast (10% default)

### 📈 **Visualizations**
- **Real-time Rankings**: Updated as you adjust weights
- **Radar Charts**: Multi-dimensional career comparison
- **Heatmaps**: Comprehensive metrics overview
- **Salary Analysis**: Distribution and comparison
- **Top Recommendations**: Personalized based on your weights

### 🎯 **Analysis Capabilities**
- **Custom Scoring**: Weight dimensions to match your priorities
- **Career Comparison**: Side-by-side analysis
- **Market Intelligence**: Salary premiums, accessibility, growth
- **Skills Analysis**: Requirements and market overlap

---

## 🐛 **Troubleshooting**

### **"Streamlit not found"**
```bash
# Verify installation
pip list | grep streamlit

# Reinstall if needed
pip install --force-reinstall streamlit
```

### **"Pandas import error"**
```bash
# Common on some systems - try virtual environment
python3 -m venv fresh_env
source fresh_env/bin/activate
pip install pandas==1.5.3 streamlit plotly
```

### **"Metrics file not found"**
```bash
# Generate the metrics data first
python3 pathfinder/metrics_engine.py

# Verify file exists
ls -la pathfinder/output/full_metrics_raw.csv
```

### **"Port 8501 already in use"**
```bash
# Use different port
streamlit run pathfinder/dashboard.py --server.port 8502

# Or kill existing process
pkill -f streamlit
```

---

## 🔄 **Alternative Options**

### **If Dependencies Won't Install:**

1. **Use the HTML Dashboard:**
   ```bash
   # Open the standalone dashboard
   open pathfinder_dashboard.html
   # Or: python3 -m http.server 8000
   ```

2. **Use the CSV Analysis:**
   ```bash
   # Load metrics in Excel/Google Sheets
   open pathfinder/output/full_metrics_raw.csv
   ```

3. **Use the Simple Scoring:**
   ```bash
   # Generate basic rankings
   python3 score_career_paths.py
   open career_path_scores.csv
   ```

---

## 🎯 **Expected Output**

When the dashboard launches successfully, you'll see:

### **📊 Main Interface**
- Interactive weight sliders in the sidebar
- Real-time career rankings table
- Top recommendation card with key metrics

### **🔍 Detailed Analysis**
- Career selection dropdown
- Multi-dimensional radar chart
- Key metrics display

### **📈 Comprehensive Visualizations**
- Metrics heatmap showing all 20+ indicators
- Salary distribution boxplot
- Market insights and recommendations

### **💡 Key Insights Section**
- Market leaders (highest salary, most accessible, most remote)
- Your custom ranking based on weight preferences
- Data-driven recommendations

---

## 📋 **Dashboard Usage Tips**

### **🎯 For Quick Analysis**
1. Keep default weights (balanced approach)
2. Check the top 3 recommendations
3. Use radar chart for detailed comparison

### **💰 For Salary Focus**
1. Increase Compensation weight to 50%+
2. Reduce other weights proportionally
3. Check salary analysis tab

### **🎓 For Career Changers**
1. Increase Accessibility weight to 40%+
2. Focus on entry-level metrics
3. Look for lower skill barriers

### **🏠 For Remote Work Priority**
1. Increase Job Market weight
2. Focus on remote work percentages
3. Check F1_remote_advantage metric

---

## 🔧 **Customization Options**

### **Modify Weights Programmatically**
Edit the default values in `pathfinder/dashboard.py`:
```python
weights['M'] = st.sidebar.slider("🧑‍💻 Job Market Weight", value=0.3)  # Change from 0.2
```

### **Add New Metrics**
Update the heatmap columns in the dashboard:
```python
heatmap_cols = [
    'M1_job_postings', 'M2_entry_pct',  # existing
    'your_new_metric'  # add new ones
]
```

### **Change Visualization Style**
Modify Plotly chart configurations:
```python
fig.update_layout(
    template="plotly_dark",  # Dark theme
    color_discrete_sequence=px.colors.qualitative.Set3  # Different colors
)
```

---

## 📞 **Support**

### **If Dashboard Won't Work:**
1. Try the virtual environment approach
2. Use the standalone HTML dashboard
3. Check that metrics data exists
4. Verify Python version (3.7+ required)

### **For Analysis Questions:**
1. Review `METRICS_ENGINE_SUMMARY.md`
2. Check `pathfinder/output/metrics_analysis.json`
3. Use the original scoring system as backup

---

**🎉 Once running, the Advanced Pathfinder Dashboard provides the most comprehensive, interactive career analysis available - with real-time customization based on your personal priorities!**

*Built with 286 real job postings • 20+ metrics • 5 analytical dimensions*