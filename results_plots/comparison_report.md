# Brain Tumor Classification - Model Comparison Report

**Generated:** 2025-10-13 16:07:02

## Executive Summary

Compared **3** models: CNN, RNN, LSTM

### Best Performing Models:
- **Accuracy:** CNN (0.8368)
- **F1 (Macro):** CNN (0.8406)

## Performance Comparison

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| CNN | 0.8368 | 0.8406 | 0.8353 |
| RNN | 0.7937 | 0.8003 | 0.7939 |
| LSTM | 0.7746 | 0.7819 | 0.7744 |

## Per-Class Performance

| Model | Glioma | Meningioma | Pituitary | No Tumor |
|-------|--------|------------|-----------|----------|
| CNN | 0.774 | 0.932 | 0.800 | 0.857 |
| RNN | 0.754 | 0.737 | 0.830 | 0.881 |
| LSTM | 0.738 | 0.713 | 0.799 | 0.876 |

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