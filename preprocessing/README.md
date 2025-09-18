# 🧹 Dataset Preprocessing Pipeline

**Combine and clean raw tactile sensor datasets for robust machine learning workflows.**

## 🚀 Quick Start (3 Steps)

1. **Prepare Data**: Place raw acquisition results in the input folder
2. **Configure**: Edit paths and parameters in `combine_full_dataset.py`
3. **Run**: `python combine_full_dataset.py` → Get unified dataset

## 📋 Requirements

**Software:**
```bash
pip install pandas numpy
```

## 🔄 Complete Workflow

1. **📥 Data Loading** - Read raw results from acquisition
2. **🧹 Data Cleaning** - Remove invalid, duplicate, or missing entries
3. **🔗 Dataset Combination** - Merge multiple trials into a single dataset
4. **💾 Save Output** - Export cleaned dataset for augmentation/training

## 📁 Output Structure

```
preprocessing/
├── full_dataset.csv           # Unified, cleaned dataset
```

## ⚙️ Key Configuration

**Edit these parameters in `combine_full_dataset.py`:**

```python
input_folder = "../acquisition/Trial_XXX/Results/"
output_file = "full_dataset.csv"
```

## 🔧 Individual Modules

| Module                  | Purpose                        | When to Use Separately           |
|-------------------------|--------------------------------|----------------------------------|
| combine_full_dataset.py | Combine and clean datasets     | Custom dataset formats           |

## 🔧 Troubleshooting

**Common Issues:**
- **Missing files**: Check input folder paths
- **Format errors**: Ensure all input CSVs match expected schema
- **Output not generated**: Check script logs for errors

**Quality Indicators:**
- No missing/NaN values in output
- Consistent column names and types

---

💡 **Pro Tip**: Validate your output dataset with a quick pandas `.describe()` or `.info()` before moving to augmentation.
