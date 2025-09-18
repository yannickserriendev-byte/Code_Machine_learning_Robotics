# 📊 Postprocessing & Analysis Pipeline

**Analyze, visualize, and refine model outputs for publication-ready results.**

## 🚀 Quick Start (3 Steps)

1. **Prepare Data**: Place predictions and labels in the input folder
2. **Configure**: Edit parameters in `postprocess_labels.py` as needed
3. **Run**: `python postprocess_labels.py` → Get analysis plots and processed labels

## 📋 Requirements

**Software:**
```bash
pip install pandas matplotlib seaborn numpy
```

## 🔄 Complete Workflow

1. **📥 Load Predictions/Labels** - Read model outputs and ground truth
2. **🔬 Analysis** - Compute metrics, compare predictions vs. labels
3. **📈 Visualization** - Generate plots for results and errors
4. **💾 Save Output** - Export processed labels and figures

## 📁 Output Structure

```
postprocessing/
├── analysis_plots/            # Figures and plots
├── processed_labels.csv       # Refined/cleaned labels
```

## ⚙️ Key Configuration

**Edit these parameters in `postprocess_labels.py`:**

```python
input_predictions = "../inference/predictions.csv"
input_labels = "../training/metrics_summary.json"
output_folder = "analysis_plots/"
```

## 🔧 Individual Modules

| Module              | Purpose                        | When to Use Separately           |
|---------------------|--------------------------------|----------------------------------|
| postprocess_labels.py| Analyze/visualize outputs      | Custom analysis, publication     |

## 🔧 Troubleshooting

**Common Issues:**
- **Missing files**: Check input paths
- **Plot errors**: Ensure matplotlib/seaborn are installed
- **Output not saved**: Check folder permissions

**Quality Indicators:**
- Plots are clear and match expected results
- No missing/NaN values in processed labels

---

💡 **Pro Tip**: Use postprocessing to create publication-ready figures and validate model performance.
