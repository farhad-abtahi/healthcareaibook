#!/usr/bin/env python3
"""
Generate notebook 8.9: Medical Scribe with Whisper
This script creates the notebook programmatically to avoid JSON escaping issues.
"""

import json
import nbformat as nbf

# Create new notebook
nb = nbf.v4.new_notebook()

# Add metadata
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    },
    "language_info": {
        "name": "python",
        "version": "3.8.0"
    },
    "accelerator": "GPU"
}

# Cell 0: Title
nb.cells.append(nbf.v4.new_markdown_cell("""# Notebook 8.9: Medical Scribe - Speech Recognition with Whisper

**Chapter 8: NLP and LLMs - Medical Documentation AI**

## Learning Objectives

By the end of this notebook, you will be able to:
1. Implement medical speech recognition using Whisper
2. Calculate Word Error Rate (WER) for transcription quality
3. **Understand why WER alone is insufficient for clinical evaluation**
4. Implement Clinical WER with error severity weighting
5. Fine-tune Whisper on medical terminology
6. Build a complete AI scribe system for clinical documentation
7. Deploy on Google Colab for accessible GPU computing

## Clinical Context

**The Documentation Burden Problem**:

Physicians spend **49% of their workday** on electronic health records (EHR):
- Average 2 hours on EHR per 1 hour of direct patient care
- Leading cause of physician burnout (62% report documentation burden)
- After-hours documentation affects work-life balance

**AI Medical Scribes**:
- Reduce documentation time by 50-70%
- Improve physician satisfaction
- Increase face-to-face patient time

**Example systems**: Nuance DAX, Suki AI, Notable Health, Abridge

---

## Setup
"""))

# Cell 1: Setup code
nb.cells.append(nbf.v4.new_code_cell("""# Check if running on Google Colab
try:
    import google.colab
    IN_COLAB = True
    print("Running on Google Colab")
except:
    IN_COLAB = False
    print("Running locally")

# Install required packages
if IN_COLAB:
    !pip install -q openai-whisper jiwer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')

print("\\nLibraries imported successfully!")
print("For production: ensure HIPAA compliance and patient consent.")
"""))

# Cell 2: WER explanation with clinical emphasis
nb.cells.append(nbf.v4.new_markdown_cell("""## Part 1: Word Error Rate (WER) - And Why It's Not Enough

**Word Error Rate (WER)** measures transcription accuracy:

$$\\text{WER} = \\frac{S + D + I}{N} \\times 100\\%$$

Where:
- **S** = Substitutions (wrong word)
- **D** = Deletions (missing word)
- **I** = Insertions (extra word)
- **N** = Total words in reference

### ⚠️ CRITICAL LIMITATION: WER Doesn't Account for Clinical Severity

**Problem**: Not all errors are equal!

**Example 1 - Low WER but DANGEROUS**:
```
Reference:  "Prescribe 10 mg of lisinopril daily"
Hypothesis: "Prescribe 100 mg of lisinopril daily"
WER: 12.5% (1 error / 8 words)
Clinical Impact: 10x overdose - potentially FATAL
```

**Example 2 - High WER but HARMLESS**:
```
Reference:  "Patient is doing very well today"
Hypothesis: "Patient is doing quite well right now"
WER: 50% (3 errors / 6 words)
Clinical Impact: Meaning preserved - no harm
```

### Clinical Error Severity Classification

**CRITICAL errors** (could cause patient harm):
- Drug names: "lisinopril" → "losartan" (different drug class!)
- Dosages: "10 mg" → "100 mg" (10x overdose)
- Negations: "no chest pain" → "chest pain" (missed MI symptom)
- Allergies: "penicillin allergy" → "penicillin" (anaphylaxis risk)

**MODERATE errors** (could affect care quality):
- Diagnoses: "atrial fibrillation" → "atrial flutter"
- Procedures: "CT scan" → "CAT scan" (acceptable synonym)
- Anatomical: "right" → "left" (critical for surgery!)

**MINOR errors** (minimal clinical impact):
- Filler words: "um", "uh", "you know"
- Synonyms: "very well" → "quite well"
- Article errors: "a" → "the"

### Solution: Clinical WER (Weighted by Severity)

$$\\text{Clinical WER} = \\frac{\\sum_{i} w_i \\times e_i}{N} \\times 100\\%$$

Where $w_i$ is the severity weight:
- Critical errors: $w = 10$
- Moderate errors: $w = 3$
- Minor errors: $w = 1$

**Goal**: Clinical WER < 5% AND no critical errors
"""))

# Cell 3: WER calculation with clinical weighting
nb.cells.append(nbf.v4.new_code_cell("""def calculate_wer(reference: str, hypothesis: str) -> Dict:
    \"\"\"Calculate Word Error Rate using edit distance.\"\"\"
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    n, m = len(ref_words), len(hyp_words)
    dp = np.zeros((n + 1, m + 1))
    ops = [[[] for _ in range(m + 1)] for _ in range(n + 1)]

    # Initialize
    for i in range(n + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = ops[i-1][0] + ['D']

    for j in range(m + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = ops[0][j-1] + ['I']

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = ops[i-1][j-1] + ['C']
            else:
                substitution = dp[i-1][j-1] + 1
                deletion = dp[i-1][j] + 1
                insertion = dp[i][j-1] + 1

                min_cost = min(substitution, deletion, insertion)
                dp[i][j] = min_cost

                if min_cost == substitution:
                    ops[i][j] = ops[i-1][j-1] + ['S']
                elif min_cost == deletion:
                    ops[i][j] = ops[i-1][j] + ['D']
                else:
                    ops[i][j] = ops[i][j-1] + ['I']

    operations = ops[n][m]
    subs = operations.count('S')
    dels = operations.count('D')
    ins = operations.count('I')
    correct = operations.count('C')

    total_errors = subs + dels + ins
    wer = (total_errors / max(n, 1)) * 100
    accuracy = (correct / max(n, 1)) * 100

    return {
        'wer': wer,
        'accuracy': accuracy,
        'substitutions': subs,
        'deletions': dels,
        'insertions': ins,
        'correct': correct,
        'total_words': n,
        'total_errors': total_errors
    }


def calculate_clinical_wer(reference: str, hypothesis: str,
                          critical_terms: List[str],
                          moderate_terms: List[str]) -> Dict:
    \"\"\"
    Calculate Clinical WER with severity weighting.

    Critical errors (weight=10): Drug names, dosages, negations
    Moderate errors (weight=3): Diagnoses, procedures
    Minor errors (weight=1): Everything else
    \"\"\"
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    # First get standard WER and alignment
    wer_result = calculate_wer(reference, hypothesis)

    # Now analyze error severity
    critical_errors = 0
    moderate_errors = 0
    minor_errors = 0

    # Simple approach: check if error involves critical/moderate terms
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)

    # Words in reference but not in hypothesis (deletions/substitutions)
    missing_words = ref_set - hyp_set

    for word in missing_words:
        if any(term in word for term in critical_terms):
            critical_errors += 1
        elif any(term in word for term in moderate_terms):
            moderate_errors += 1
        else:
            minor_errors += 1

    # Extra words (insertions/substitutions)
    extra_words = hyp_set - ref_set
    for word in extra_words:
        if any(term in word for term in critical_terms):
            critical_errors += 1
        elif any(term in word for term in moderate_terms):
            moderate_errors += 1
        else:
            minor_errors += 1

    # Calculate weighted WER
    weighted_errors = (critical_errors * 10 +
                      moderate_errors * 3 +
                      minor_errors * 1)
    clinical_wer = (weighted_errors / max(len(ref_words), 1)) * 100

    return {
        'standard_wer': wer_result['wer'],
        'clinical_wer': clinical_wer,
        'critical_errors': critical_errors,
        'moderate_errors': moderate_errors,
        'minor_errors': minor_errors,
        'clinically_safe': critical_errors == 0
    }


# Define critical and moderate medical terms
CRITICAL_TERMS = [
    # Drugs that sound similar
    'lisinopril', 'losartan', 'metformin', 'metoprolol',
    # Dosages
    '10', '100', 'mg', 'mcg', 'ml',
    # Negations
    'no', 'not', 'without', 'denies',
    # Allergies
    'allergy', 'allergic', 'penicillin'
]

MODERATE_TERMS = [
    'hypertension', 'diabetes', 'pneumonia', 'copd',
    'right', 'left', 'bilateral',
    'ct', 'mri', 'x-ray'
]

print("✓ WER calculation functions ready")
print("✓ Clinical severity weighting implemented")
"""))

# Cell 4: Demonstrate clinical WER
nb.cells.append(nbf.v4.new_code_cell("""print("="*80)
print("CLINICAL WER DEMONSTRATION - Why Standard WER Is Insufficient")
print("="*80)

dangerous_examples = [
    {
        'name': 'CRITICAL: Drug dose error (10x overdose)',
        'reference': 'Prescribe 10 mg of lisinopril daily',
        'hypothesis': 'Prescribe 100 mg of lisinopril daily',
        'expected': 'Low standard WER, HIGH clinical WER, UNSAFE'
    },
    {
        'name': 'CRITICAL: Negation error (missed symptom)',
        'reference': 'Patient denies chest pain',
        'hypothesis': 'Patient has chest pain',
        'expected': 'Low standard WER, HIGH clinical WER, UNSAFE'
    },
    {
        'name': 'CRITICAL: Drug name confusion',
        'reference': 'Continue lisinopril for hypertension',
        'hypothesis': 'Continue losartan for hypertension',
        'expected': 'Low standard WER, but WRONG DRUG CLASS'
    },
    {
        'name': 'MINOR: Filler words (harmless)',
        'reference': 'Patient is doing very well today',
        'hypothesis': 'Patient is doing quite well right now',
        'expected': 'High standard WER, low clinical impact, SAFE'
    }
]

for example in dangerous_examples:
    print(f"\\n{'-'*80}")
    print(f"Example: {example['name']}")
    print(f"{'-'*80}")
    print(f"Reference:  '{example['reference']}'")
    print(f"Hypothesis: '{example['hypothesis']}'")

    # Standard WER
    standard = calculate_wer(example['reference'], example['hypothesis'])

    # Clinical WER
    clinical = calculate_clinical_wer(
        example['reference'],
        example['hypothesis'],
        CRITICAL_TERMS,
        MODERATE_TERMS
    )

    print(f"\\nStandard WER: {standard['wer']:.1f}%")
    print(f"Clinical WER: {clinical['clinical_wer']:.1f}%")
    print(f"Critical errors: {clinical['critical_errors']}")
    print(f"Moderate errors: {clinical['moderate_errors']}")
    print(f"Minor errors: {clinical['minor_errors']}")
    print(f"Clinically safe: {'✓ YES' if clinical['clinically_safe'] else '✗ NO - UNSAFE'}")
    print(f"Expected: {example['expected']}")

print("\\n" + "="*80)
print("KEY INSIGHT: Standard WER Can Be Misleading!")
print("="*80)
print("""
A system with 5% standard WER might seem acceptable, but if those 5% errors
include critical medical terms, it could be DANGEROUS.

Always use Clinical WER and check for critical errors before deployment.

Recommended thresholds:
  - Standard WER < 10% (general quality)
  - Clinical WER < 5% (weighted by severity)
  - Critical errors = 0 (MANDATORY for clinical safety)
""")
"""))

# Save the notebook
with open('notebook_8_9_medical_scribe.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
