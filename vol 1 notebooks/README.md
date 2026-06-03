# Volume 1: Foundations and Technical Principles

_Artificial Intelligence in Healthcare, Volume 1_

This directory holds the **44 hands-on Jupyter notebooks** that accompany Volume 1, organized into one folder per chapter (`chapter_04` through `chapter_11`). Chapters 1 to 3 are conceptual and have no notebooks.

Every notebook is self-contained, includes explanatory markdown and exercises, and runs in Google Colab with no local setup.

## Getting started

**Google Colab (no setup):** open any notebook in the browser using this link pattern (note the spaces in the folder name are written as `%20` in URLs):

```
https://colab.research.google.com/github/farhad-abtahi/healthcareaibook/blob/main/vol%201%20notebooks/chapter_04/notebook_4_1_data_quality_assessment.ipynb
```

**Local:**

```bash
git clone https://github.com/farhad-abtahi/healthcareaibook.git
cd "healthcareaibook/vol 1 notebooks"
pip install -r requirements.txt
jupyter notebook
```

## Naming convention

Files follow `notebook_X_Y_title.ipynb`, where `X` is the chapter (4 to 11) and `Y` is the sequence within the chapter.

## Notebook index

### Chapter 4: Data Preparation

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`4.1`](chapter_04/notebook_4_1_data_quality_assessment.ipynb) | Data Quality Assessment | Missing value analysis, outlier detection, data profiling | Foundation for all journeys |
| [`4.2`](chapter_04/notebook_4_2_missing_data_handling.ipynb) | Missing Data Handling | MCAR/MAR/MNAR analysis, imputation (mean, KNN, MICE) | All journeys with EHR data |
| [`4.3`](chapter_04/notebook_4_3_feature_engineering.ipynb) | Feature Engineering | Clinical feature creation, temporal and domain-specific features | Marcus, David |
| [`4.4`](chapter_04/notebook_4_4_cross_validation.ipynb) | Cross-Validation Strategies | K-fold, stratified, GroupKFold, TimeSeriesSplit, nested CV | All journeys |
| [`4.5`](chapter_04/notebook_4_5_data_leakage.ipynb) | Data Leakage Detection | Target, temporal, and preprocessing leakage detection | Critical for all journeys |

### Chapter 5: Evaluation Metrics

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`5.1`](chapter_05/notebook_5_1_classification_metrics.ipynb) | Classification Metrics | Confusion matrix, sensitivity, specificity, PPV, NPV, F1 | All classification tasks |
| [`5.2`](chapter_05/notebook_5_2_roc_pr_curves.ipynb) | ROC and PR Curves | AUROC, AUPRC, threshold selection, operating points | Elena, Jamal |
| [`5.3`](chapter_05/notebook_5_3_cvd_risk_calibration.ipynb) | Calibration Analysis (CVD Risk) | Reliability diagrams, Brier score, Platt scaling | David |
| [`5.4`](chapter_05/notebook_5_4_regression_metrics.ipynb) | Regression Metrics | MSE, MAE, R-squared, clinical error analysis | Continuous outcome prediction |
| [`5.5`](chapter_05/notebook_5_5_segmentation_metrics.ipynb) | Segmentation Metrics | Dice coefficient, IoU, pixel accuracy, boundary metrics | Medical image segmentation |
| [`5.6`](chapter_05/notebook_5_6_fairness_metrics.ipynb) | Fairness Metrics Preview | Subgroup AUROC, demographic stratification, disparity detection | Priya, all journeys |

### Chapter 6: Medical Imaging and Computer Vision

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`6.1`](chapter_06/notebook_6_1_cnn_fundamentals.ipynb) | CNN Fundamentals | Convolution, pooling, architecture design, PyTorch | Foundation for imaging |
| [`6.2`](chapter_06/notebook_6_2_melanoma_classification_fairness.ipynb) | Melanoma Classification and Fairness | Transfer learning, fine-tuning, fairness evaluation | Priya |
| [`6.3`](chapter_06/notebook_6_3_preprocessing.ipynb) | Image Preprocessing | Normalization, resizing, CT windowing, augmentation | All imaging journeys |
| [`6.4`](chapter_06/notebook_6_4_augmentation.ipynb) | Data Augmentation | Geometric and color transforms, MixUp, CutMix | Elena, Jamal, Priya |
| [`6.5`](chapter_06/notebook_6_5_lung_nodule_detection.ipynb) | Lung Nodule Detection | Object detection, anchor boxes, false positive reduction | Jamal |
| [`6.6`](chapter_06/notebook_6_6_brain_tumor_segmentation_unet.ipynb) | Brain Tumor Segmentation (U-Net) | U-Net, encoder-decoder, skip connections, multi-class | Elena |
| [`6.7`](chapter_06/notebook_6_7_gradcam.ipynb) | GradCAM Visualization | Gradient-weighted class activation maps, attention | All imaging (interpretability) |

### Chapter 7: Time Series Analysis

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`7.1`](chapter_07/notebook_7_1_signal_preprocessing.ipynb) | Signal Preprocessing | Filtering, resampling, artifact removal, baseline correction | Marcus, Yuki |
| [`7.2`](chapter_07/notebook_7_2_feature_extraction.ipynb) | Feature Extraction | Statistical and frequency-domain features, wavelets, HRV | Yuki, Marcus |
| [`7.3`](chapter_07/notebook_7_3_sepsis_prediction_xgboost.ipynb) | Sepsis Prediction with XGBoost | Gradient boosting, feature importance, early warning scores | Marcus |
| [`7.4`](chapter_07/notebook_7_4_ecg_classification_1d_cnn.ipynb) | ECG Classification with 1D CNN | 1D convolutions, arrhythmia detection, multi-lead ECG | Yuki |
| [`7.5`](chapter_07/notebook_7_5_forecasting.ipynb) | Time Series Forecasting | ARIMA, Prophet, sequence models, vital sign prediction | ICU monitoring |
| [`7.6`](chapter_07/notebook_7_6_realtime_processing.ipynb) | Real-time Processing | Streaming data, sliding windows, online learning, alerts | Marcus |

### Chapter 8: Natural Language Processing and LLMs

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`8.1`](chapter_08/notebook_8_1_text_preprocessing.ipynb) | Text Preprocessing | Tokenization, medical abbreviations, negation detection | Aisha |
| [`8.2`](chapter_08/notebook_8_2_ner.ipynb) | Named Entity Recognition | Clinical NER, medication extraction, spaCy and Transformers | Aisha |
| [`8.3`](chapter_08/notebook_8_3_bert_classification.ipynb) | BERT for Classification | Fine-tuning BERT, clinical text classification | All text applications |
| [`8.4`](chapter_08/notebook_8_4_llm_prompting.ipynb) | LLM Prompting | Prompt engineering, few-shot learning, chain-of-thought | Aisha |
| [`8.5`](chapter_08/notebook_8_5_rag.ipynb) | Retrieval-Augmented Generation | RAG architecture, vector databases, knowledge retrieval | Clinical decision support |
| [`8.6`](chapter_08/notebook_8_6_summarization.ipynb) | Clinical Summarization | Abstractive summarization, discharge summaries, progress notes | Aisha |
| [`8.7`](chapter_08/notebook_8_7_hallucination_detection.ipynb) | Hallucination Detection | Factual grounding, citation verification, confidence calibration | All LLM applications |
| [`8.8`](chapter_08/notebook_8_8_llm_differential_diagnosis.ipynb) | LLM Differential Diagnosis | LLM reasoning, diagnostic support, uncertainty quantification | Clinical decision support |

> **Bonus (optional): 8.9 Medical Scribe with Whisper.** Provided as a generator script (`chapter_08/generate_notebook_8_9.py`) and a summary (`chapter_08/notebook_8_9_summary.md`). It covers speech-to-text clinical transcription with Whisper and a clinically weighted word error rate. It is not counted among the 44 core notebooks.

### Chapter 9: Fairness and Bias

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`9.1`](chapter_09/notebook_9_1_dataset_bias_measurement.ipynb) | Dataset Bias Measurement | Representation analysis, label quality, measurement bias | Foundation for fairness |
| [`9.2`](chapter_09/notebook_9_2_fairness_metrics.ipynb) | Fairness Metrics | Demographic parity, equalized odds, equal opportunity | All journeys |
| [`9.3`](chapter_09/notebook_9_3_bias_mitigation.ipynb) | Bias Mitigation | Resampling, SMOTE, fairness constraints, threshold optimization | Priya |
| [`9.4`](chapter_09/notebook_9_4_priyas_journey.ipynb) | Priya's Journey | Complete fairness case study: biased model, harm analysis, correction | Priya (Journey 5) |
| [`9.5`](chapter_09/notebook_9_5_intersectional_fairness.ipynb) | Intersectional Fairness | Multi-attribute analysis, compounding bias, targeted mitigation | Vulnerable populations |

### Chapter 10: Interpretability

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`10.1`](chapter_10/notebook_10_1_shap_feature_importance.ipynb) | SHAP Feature Importance | SHAP values, dependence plots, global and local explanations | David |
| [`10.2`](chapter_10/notebook_10_2_lime_explanations.ipynb) | LIME Explanations | Local surrogate models, tabular, image, and text LIME | All modalities |
| [`10.3`](chapter_10/notebook_10_3_gradcam_imaging.ipynb) | GradCAM for Imaging | Gradient visualization, attention maps, clinical validation | Elena, Jamal, Priya |
| [`10.4`](chapter_10/notebook_10_4_counterfactual_explanations.ipynb) | Counterfactual Explanations | What-if analysis, actionable recourse, minimal changes | Clinical decision support |

### Chapter 11: Privacy and Security

| Notebook | Title | Key skills | Clinical journey |
|---|---|---|---|
| [`11.1`](chapter_11/notebook_11_1_differential_privacy.ipynb) | Differential Privacy | Laplace mechanism, DP-SGD, privacy composition, Opacus | All privacy-sensitive applications |
| [`11.2`](chapter_11/notebook_11_2_federated_learning.ipynb) | Federated Learning | FedAvg, non-IID handling, secure aggregation | Multi-hospital collaboration |
| [`11.3`](chapter_11/notebook_11_3_adversarial_robustness.ipynb) | Adversarial Robustness | FGSM and PGD attacks, adversarial training, certified defenses | All deployed models |

## Requirements

Dependencies are pinned in [`requirements.txt`](requirements.txt) in this folder. The stack includes `numpy`, `pandas`, `scikit-learn`, `xgboost`, `torch`, `torchvision`, `scikit-image`, `transformers`, `spacy`, `shap`, `lime`, `opacus`, and `pywavelets`.

## About this volume

Volume 1 establishes the technical foundation for clinical AI: machine learning principles, healthcare data preparation, evaluation methodology, domain-specific methods for imaging, time series, and natural language processing, and responsible practice covering fairness, interpretability, privacy, and security.

ISBN 978-91-991639-0-1. First Edition, 2026. Authors: Farhad Abtahi and Mehdi Astaraki.

Back to the [repository home](https://github.com/farhad-abtahi/healthcareaibook).
