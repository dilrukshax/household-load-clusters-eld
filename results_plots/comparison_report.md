# Brain Tumor Classification - Model Comparison Report

**Generated:** 2025-10-12 19:55:17

## Executive Summary

Compared **3** models: CNN, RNN, LSTM

### Best Performing Models:
- **Accuracy:** RNN (0.5953)
- **F1 (Macro):** RNN (0.5850)

## Performance Comparison

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| CNN | 0.2976 | 0.1147 | 0.1365 |
| RNN | 0.5953 | 0.5850 | 0.5803 |
| LSTM | 0.5230 | 0.5093 | 0.4975 |

## Per-Class Performance

| Model | Glioma | Meningioma | Pituitary | No Tumor |
|-------|--------|------------|-----------|----------|
| CNN | 0.459 | 0.000 | 0.000 | 0.000 |
| RNN | 0.598 | 0.390 | 0.650 | 0.701 |
| LSTM | 0.551 | 0.400 | 0.388 | 0.699 |

## Generated Visualizations

1. `accuracy_f1_comparison.png` - Overall performance metrics
2. `per_class_f1_comparison.png` - Per-class F1 scores
3. `radar_chart_per_class_f1.png` - Radar chart comparison
4. `confusion_matrices_comparison.png` - Side-by-side confusion matrices
5. `training_curves_comparison.png` - Training history overlay
6. `sample_validation_images.png` - Sample validation images

## Files Generated

- `results_summary.csv` - Main results table
- `comprehensive_summary.csv` - Detailed statistics
- `sample_predictions_comparison.csv` - Per-image predictions
- `results_plots/` - All visualization files

---

*End of Report*