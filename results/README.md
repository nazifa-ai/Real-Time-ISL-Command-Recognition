# Results

This directory intentionally contains **no pre-filled model performance numbers**.

Run the reproducible pipeline after the INCLUDE-50 keypoints are available:

```bash
python run.py preprocess --protocol session_disjoint
python run.py train --protocol session_disjoint
python run.py evaluate --protocol session_disjoint
python run.py robustness --protocol session_disjoint
python run.py threshold --protocol session_disjoint
```

The generated CSV/JSON/plots are the results to use in the final report.
