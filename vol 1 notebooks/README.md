# Notebooks for AI in Healthcare: Volume 1

This directory contains 47 hands-on Jupyter notebooks that accompany the textbook chapters. Each notebook provides practical implementations, exercises, and real-world examples to reinforce the concepts covered in the text.

## 📚 How to Use These Notebooks

### Getting Started

1. **Online (Recommended for Beginners)**:
   - Open notebooks directly in [Google Colab](https://colab.research.google.com/) - no setup required
   - Click the "Open in Colab" badge at the top of each notebook
   - All dependencies are pre-installed

2. **Local Environment**:
   - Clone this repository
   - Install dependencies: `pip install -r requirements.txt`
   - Launch Jupyter: `jupyter notebook`

### Learning Path

- **For Clinicians**: Focus on notebooks 4.1, 5.1, 5.3, 9.4, and 10.1-10.4 (interpretability)
- **For Data Scientists**: Complete all notebooks sequentially
- **For Students**: Work through all notebooks with exercises
- **For Administrators**: Review notebooks 5.3, 9.2, 9.4, 11.1-11.3

---

## 📓 Notebook Inventory by Chapter

### Chapter 4: Data Preparation and Cross-Validation (5 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 4.1 | Data Quality Assessment in Healthcare | Completeness, outliers, consistency checks |
| 4.2 | Missing Data Handling in Healthcare | MCAR, MAR, MNAR; imputation strategies |
| 4.3 | Feature Engineering for Healthcare ML | Temporal features, domain-specific features |
| 4.4 | Cross-Validation Strategies for Healthcare ML | K-fold, stratified, time-series splitting |
| 4.5 | Data Leakage Detection and Prevention | Temporal leakage, target leakage |

---

### Chapter 5: Evaluation Metrics and Model Performance (6 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 5.1 | Classification Metrics Deep Dive | Sensitivity, specificity, PPV, NPV, F1 |
| 5.2 | ROC and Precision-Recall Curves | AUROC, AUPRC, threshold selection |
| 5.3 | CVD Risk Prediction and Calibration | Calibration curves, Brier score |
| 5.4 | Regression Metrics | MAE, RMSE, R², residual analysis |
| 5.5 | Segmentation Metrics | Dice coefficient, IoU, Hausdorff distance |
| 5.6 | Fairness Metrics | Demographic parity, equal opportunity |

**Key Journey**: Notebook 5.3 demonstrates David's cardiovascular risk prediction journey from Chapter 3.

---

### Chapter 6: Medical Imaging and Computer Vision (7 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 6.1 | CNN Fundamentals | Convolutional layers, pooling, architectures |
| 6.2 | Melanoma Classification with Fairness Analysis | **Priya's Journey** - skin tone bias |
| 6.3 | Image Preprocessing | Normalization, resizing, windowing |
| 6.4 | Data Augmentation | Rotation, flipping, color jittering |
| 6.5 | Lung Nodule Detection with Object Detection | **Jamal's Journey** - YOLO, Faster R-CNN |
| 6.6 | Brain Tumor Segmentation with U-Net | **Elena's Journey** - 3D segmentation |
| 6.7 | GradCAM Visualization | Attention maps, model debugging |
| 6.8 | Transfer Learning for Medical Imaging | Fine-tuning, feature extraction |

**Key Journeys**:
- Notebook 6.2 → Journey 5 (Priya's melanoma)
- Notebook 6.5 → Journey 3 (Jamal's lung nodules)
- Notebook 6.6 → Journey 4 (Elena's brain tumor)

---

### Chapter 7: Time Series and Physiological Signals (6 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 7.1 | Signal Preprocessing for Time Series Analysis | Filtering, artifact removal, resampling |
| 7.2 | Time Series Feature Extraction for Clinical Signals | Statistical features, frequency domain |
| 7.3 | Sepsis Prediction from Time Series Data | **Marcus's Journey** - XGBoost, SHAP |
| 7.4 | ECG Signal Classification for AFib Detection | **Yuki's Journey** - 1D CNN for ECG |
| 7.5 | Time Series Forecasting for Clinical Applications | ARIMA, LSTM, Prophet |
| 7.6 | Real-Time Signal Processing for Clinical Monitoring | Streaming data, online learning |
| 7.7 | RNN and LSTM Architectures | Sequence models, vanishing gradients |

**Key Journeys**:
- Notebook 7.3 → Journey 1 (Marcus's sepsis)
- Notebook 7.4 → Journey 2 (Yuki's AFib)

---

### Chapter 8: Natural Language Processing and LLMs (8 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 8.1 | Clinical Text Preprocessing | Tokenization, negation detection |
| 8.2 | Named Entity Recognition (NER) | MedCAT, spaCy, medication extraction |
| 8.3 | BERT for Clinical Text Classification | Fine-tuning, sequence classification |
| 8.4 | LLM Prompting Strategies | Zero-shot, few-shot, chain-of-thought |
| 8.5 | Retrieval-Augmented Generation (RAG) for Clinical Decision Support | Vector databases, semantic search |
| 8.6 | Clinical Text Summarization | Abstractive, extractive summarization |
| 8.7 | Hallucination Detection in Clinical LLMs | Factuality checking, confidence estimation |
| 8.8 | LLM-Based Differential Diagnosis System | **Aisha's Journey** - GPT-4, Claude |

**Key Journey**: Notebook 8.8 → Journey 7 (Aisha's differential diagnosis)

---

### Chapter 9: Fairness and Bias in Healthcare AI (5 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 9.1 | Measuring Dataset Bias | Representation analysis, demographic distributions |
| 9.2 | Fairness Metrics Implementation | All four fairness metrics, Chouldechova's theorem |
| 9.3 | Bias Mitigation Strategies | Resampling, reweighting, threshold optimization |
| 9.4 | Priya's Journey - When Bias Becomes Personal | **Complete Journey 5 walkthrough** |
| 9.5 | Intersectional Fairness | Multiple protected attributes, compound bias |

**Key Journey**: Notebook 9.4 is the complete implementation of Priya's melanoma journey from Chapter 3, demonstrating skin tone bias.

---

### Chapter 10: Interpretability and Explainability (4 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 10.1 | Feature Importance and SHAP Values | SHAP waterfall plots, force plots, summary plots |
| 10.2 | LIME - Local Interpretable Model-agnostic Explanations | Tabular, image, text explanations |
| 10.3 | GradCAM for Medical Imaging | Attention visualization, model debugging |
| 10.4 | Counterfactual Explanations - "What If" Clinical Scenarios | DiCE, actionable recourse |

**All notebooks**: Demonstrate interpretability methods critical for clinical trust and regulatory compliance.

---

### Chapter 11: Privacy, Security, and Trustworthy AI (3 notebooks)

| Notebook | Title | Topics Covered |
|----------|-------|----------------|
| 11.1 | Differential Privacy for Healthcare AI | ε-DP, privacy-utility trade-off |
| 11.2 | Federated Learning for Healthcare | FedAvg, multi-hospital collaboration |
| 11.3 | Adversarial Robustness | FGSM, PGD attacks, defenses |

**All notebooks**: Essential for deploying trustworthy AI in healthcare settings.

---

## 📊 Total Notebooks: 44

**By Chapter**:
- Chapter 4: 5 notebooks (Data Preparation)
- Chapter 5: 6 notebooks (Evaluation Metrics)
- Chapter 6: 7 notebooks (Medical Imaging)
- Chapter 7: 6 notebooks (Time Series)
- Chapter 8: 8 notebooks (NLP & LLMs)
- Chapter 9: 5 notebooks (Fairness & Bias)
- Chapter 10: 4 notebooks (Interpretability)
- Chapter 11: 3 notebooks (Privacy & Security)

**Note**: Chapters 0-3 do not have separate notebooks. Chapter 3's journeys are implemented in notebooks throughout later chapters (see "Key Journeys" markers above).

---

## 🔗 Accessing Notebooks

All notebooks are available in the book's GitHub repository:

**[GitHub Repository - AI in Healthcare Volume 1 Notebooks]** *(see Chapter 0 for link)*

### Quick Start with Google Colab

Each notebook includes a "Open in Colab" badge. Simply click it to launch the notebook in Google Colab with all dependencies pre-configured.

### Running Locally

```bash
# Clone the repository
git clone https://github.com/[YOUR-USERNAME]/AI-in-healthcare-book

# Navigate to notebooks directory
cd AI-in-healthcare-book/notebooks

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

**Dependencies**: All required packages are listed in `notebooks/requirements.txt`. Major dependencies include:
- NumPy, Pandas, Matplotlib, Seaborn (core scientific computing)
- Scikit-learn, XGBoost (machine learning)
- PyTorch, Torchvision (deep learning)
- Transformers, OpenAI (NLP and LLMs)
- SHAP, LIME (interpretability)
- Scikit-image, pydicom (medical imaging)

---

## 📋 Notebook Structure

Each notebook follows a consistent structure:

1. **Header**: Title, chapter reference, learning objectives
2. **Setup**: Imports, configuration, data loading
3. **Conceptual Introduction**: Brief theory recap from text chapter
4. **Hands-On Implementation**: Step-by-step code with explanations
5. **Visualizations**: Charts, plots, and interactive elements
6. **Clinical Context**: Real-world implications
7. **Exercises**: Practice problems (solutions provided)
8. **Summary**: Key takeaways and next steps

---

## 🎯 Learning Objectives

By completing these notebooks, you will:

- ✅ Implement ML pipelines from data preparation through deployment
- ✅ Build and evaluate models using healthcare-appropriate metrics
- ✅ Create CNNs for medical imaging tasks
- ✅ Process time-series physiological signals
- ✅ Apply NLP and LLMs to clinical text
- ✅ Measure and mitigate algorithmic bias
- ✅ Generate model explanations for clinical trust
- ✅ Implement privacy-preserving techniques

---

## ⚠️ Important Notes

### Duplicate Files

Some chapters have duplicate notebook files with slightly different names:
- `notebook_4_1_data_quality.ipynb` vs `notebook_4_1_data_quality_assessment.ipynb`
- `notebook_4_2_missing_data.ipynb` vs `notebook_4_2_missing_data_handling.ipynb`

**Use the files with full descriptive names** (e.g., `*_assessment.ipynb`, `*_handling.ipynb`) as they contain the complete, updated content.

### Data Requirements

- **Synthetic Data**: Most notebooks use synthetic data generated within the notebook
- **Public Datasets**: Some notebooks use public datasets (MIMIC, PhysioNet) - instructions provided
- **No PHI**: All notebooks are HIPAA-compliant with no real patient data

### Computational Requirements

- **Lightweight** (Chapters 4-5, 9-11): Run on CPU, minimal RAM
- **Moderate** (Chapters 7-8): GPU recommended but not required
- **Heavy** (Chapter 6): GPU strongly recommended for imaging tasks

Google Colab provides free GPU access, making all notebooks accessible without local GPU hardware.

---

## 🆘 Getting Help

- **Issues**: Report bugs or request clarifications via GitHub Issues
- **Questions**: Use GitHub Discussions for conceptual questions
- **Contributions**: Pull requests welcome for improvements or corrections

---

## 📚 Integration with Text Chapters

Notebooks are designed to be used **alongside** the text chapters, not as replacements:

1. **Read the text chapter first** to understand concepts
2. **Work through the corresponding notebooks** for hands-on practice
3. **Complete exercises** to reinforce learning
4. **Refer back to text** for deeper theoretical understanding

The text provides the "why," the notebooks provide the "how."

---

## 🔄 Updates and Maintenance

This notebook collection is actively maintained. Check the repository for:
- Bug fixes
- Updated dependencies
- New examples
- Community contributions

**Last Updated**: 2025-11-05
**Total Notebooks**: 44 (all with "Open in Colab" badges)
**Format**: Jupyter Notebook (.ipynb)
**Status**: ✅ 44 complete and functional
**License**: See repository for license information

---

## 🎊 Notebook Completion Status

### All Notebooks Functional (44/44) ✅

All notebooks are complete with working code, detailed explanations, and visualizations. Ready for immediate use!

**Removed duplicates/placeholders** (Nov 2025):
- Removed `chapter_04/notebook_4_1_data_quality.ipynb` (older duplicate - use `notebook_4_1_data_quality_assessment.ipynb`)
- Removed `chapter_04/notebook_4_2_missing_data.ipynb` (older duplicate - use `notebook_4_2_missing_data_handling.ipynb`)
- Removed `chapter_08/notebook_8_9_medical_scribe.ipynb` (empty placeholder)

### Common Imports Across Notebooks
- **NumPy, Pandas**: 43/47 notebooks (91%)
- **Matplotlib, Seaborn**: 43/47 notebooks (91%)
- **Scikit-learn**: 35/47 notebooks (74%)
- **PyTorch**: 15/47 notebooks (32%, primarily imaging/NLP)
- **Transformers**: 8/47 notebooks (17%, NLP chapters)

All dependencies are specified in `notebooks/requirements.txt`.

---

**Happy Learning!** 🚀

*These notebooks accompany "AI in Healthcare: Volume 1 - Foundations"*
