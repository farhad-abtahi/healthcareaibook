# Artificial Intelligence in Healthcare

### Companion code and notebooks for the three-volume book series

This repository holds the companion Jupyter notebooks, exercises, and governance templates for the **Artificial Intelligence in Healthcare** series, a graduate-level set of textbooks that moves from how clinical AI works, to how it reaches patients safely, to how it scales across specialized contexts.

- Companion website: [aiinhealthcarebook.com](https://aiinhealthcarebook.com)
- Volume 1 on Amazon: ISBN 978-91-991639-0-1
- Course adoption: Karolinska Institutet and the University of Bern (MSc in AI in Medicine)

---

## The series

| Volume | Subtitle | Theme | Status |
|---|---|---|---|
| **1** | Foundations and Technical Principles | How AI works | Available now |
| **2** | Implementation, Governance, and Clinical Translation | How AI deploys | Coming soon |
| **3** | Specialized Domains and Emerging Frontiers | How AI scales | Coming soon |

Each volume can be read on its own, but they are designed to build on one another. The seven clinical journeys introduced in Volume 1 recur across all three.

---

## What is in this repository

```
healthcareaibook/
├── README.md              <- you are here
├── requirements.txt       <- shared Python dependencies
├── volume-1/              <- Foundations: 44 notebooks (Chapters 4 to 11)
│   └── README.md          <- full Volume 1 notebook index
└── volume-2/              <- Clinical Translation: 5 notebooks + governance templates
    └── README.md          <- full Volume 2 notebook index
```

- **[Volume 1 notebooks](volume-1/README.md)**: 44 hands-on notebooks covering data preparation, evaluation, medical imaging, time series, clinical NLP and LLMs, fairness, interpretability, and privacy.
- **[Volume 2 materials](volume-2/README.md)**: 5 notebooks plus editable regulatory checklists, a lifecycle crosswalk, and an implementation playbook.

---

## Getting started

### Run in Google Colab (no setup)

Open any notebook directly in the browser. The Colab link follows this pattern:

```
https://colab.research.google.com/github/farhad-abtahi/healthcareaibook/blob/main/<path-to-notebook>
```

For example, the first Volume 1 notebook:

```
https://colab.research.google.com/github/farhad-abtahi/healthcareaibook/blob/main/volume-1/notebook_4_1_data_quality_assessment.ipynb
```

### Run locally

```bash
git clone https://github.com/farhad-abtahi/healthcareaibook.git
cd healthcareaibook
pip install -r requirements.txt
jupyter notebook
```

A recent Python (3.10 or newer) is recommended. Notebooks that train deep learning models benefit from a GPU but will run on CPU for the provided examples.

---

## The seven clinical journeys

Seven patient cases recur throughout the series, connecting technical methods to patient care:

| Patient | Condition | AI application |
|---|---|---|
| Marcus | Sepsis | Early warning prediction |
| Elena | Brain tumor | MRI segmentation |
| Yuki | Atrial fibrillation | ECG and wearable detection |
| Jamal | Lung cancer | CT nodule detection |
| Priya | Melanoma | Dermoscopy classification and fairness |
| Aisha | Documentation | Clinical NLP and summarization |
| David | Heart failure | Readmission prediction |

> The seven journeys are fictional educational constructs created for teaching. They do not represent real patients or real cases, and the datasets used in the notebooks are public or synthetic.

---

## How to use these materials in a course

Each chapter pairs a reading with one or more notebooks that can serve as lab assignments. The series supports a one-semester course per volume, or a two-course sequence across Volumes 1 and 2, with the seven journeys as running case discussions. A maximum student workload of about 60 hours per volume is assumed.

---

## Citation

If you use these materials, please cite the book:

```bibtex
@book{abtahi2026aihealthcare_vol1,
  title     = {Artificial Intelligence in Healthcare, Volume 1: Foundations and Technical Principles},
  author    = {Abtahi, Farhad and Astaraki, Mehdi},
  year      = {2026},
  edition   = {First},
  publisher = {Fivation AB},
  address   = {Stockholm, Sweden},
  isbn      = {978-91-991639-0-1}
}
```

---

## License and use

The textbooks are copyrighted. The companion code in this repository is provided for educational use alongside the books. Please see the `LICENSE` file for the terms that apply to the code, and do not redistribute book text or figures without permission.

The notebooks reference external libraries and datasets that carry their own licenses. Regulatory and clinical guidance in Volume 2 may change after publication; always verify current requirements against authoritative sources.

---

## Authors

- **Farhad Abtahi**, lead author of the series. Co-author of Volumes 1 and 2, author of Volume 3.
- **Mehdi Astaraki**, co-author of Volume 1.
- **Fernando Seoane**, co-author of Volume 2.

Questions and corrections are welcome through the repository issue tracker.
