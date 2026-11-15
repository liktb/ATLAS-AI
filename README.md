### 1. README.md

```markdown
# ATLAS: Automated Tracking of Land, Agriculture & Sprawl

A production-grade geospatial AI monitoring pipeline built using PyTorch, Torchvision (**ResNet-18**), and accelerated locally via NVIDIA CUDA (with automated CPU fallback). Designed for high-accuracy land-use multi-label classification, temporal change detection, and predictive urban sprawl/water scarcity risk modeling on satellite imagery.

---

##  Model Performance & Evaluation Metrics

Evaluated on the BigEarthNet v2.0 (reBEN) held-out test dataset:
- **Macro Mean Average Precision (mAP):** `66.53%`
- **Micro Mean Average Precision (mAP):** `83.28%`
- **Macro F1-Score:** `59.33%`
- **Micro F1-Score:** `72.65%`
- **Hamming Loss (Error Rate):** `7.90%`

---

##  Regional Case Study & Sample Validation
The repository includes un-trained sample historical and modern satellite tiles located in the `sample_tiles/` directory (e.g., Ramhan Island, Abu Dhabi) to test temporal inference immediately:
- **Baseline Tile (`ramhan_2016.png`):** Captures natural coastal sabkha and early earth-moving operations.
- **Target Tile (`ramhan_2026.png`):** Captures the completed luxury waterfront canal and residential megaproject.

---

##  Project Structure

```text
Atlas-AI/
│
├── models/
│   └── resnet18_bigearthnet_v2.pth   # Trained multi-label model weights
├── sample_tiles/
│   ├── ramhan_2016.png               # Sample historical validation tile
│   └── ramhan_2026.png               # Sample modern validation tile
├── src/
│   ├── dataset.py                    # Optimized multi-label data loader pipeline
│   ├── train.py                      # Mixed-precision training script
│   ├── test.py                       # Evaluation metrics & multi-label report generator
│   └── inference.py                  # Single-tile prediction & risk scoring script
├── app.py                            # Streamlit geospatial command-center interface
├── run_analysis.py                   # Standalone CLI mathematical telemetry & simulation script
├── requirements.txt                  # Locked python package dependencies
├── .gitignore                        # Git exclusion rules for large checkpoints
└── README.md                         # Project documentation & metrics summary

```

---

##  How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Run Standalone CLI Telemetry Analysis

To execute the model locally on two sample or custom image tiles without launching the web application:

```bash
python run_analysis.py

```

### 3. Launch the Streamlit Geospatial Command-Center Dashboard

To launch the full interactive web application featuring split-viewports, swipe sliders, and predictive simulation curves:

```bash
streamlit run app.py

```

---

##  Authors & Team

* **Pipeline Architecture, Training & Execution:** [Ryan Dinto](https://github.com/liktb)
* **Dataset Curation & Data Splitting Lead:** [Aaryan Kalra](https://github.com/aaryankalra1510-lgtm)
* **Application Integration & Streamlit Demo Interface Lead:** [Devansh Sharma](https://github.com/devanshs1802-glitch)

```
