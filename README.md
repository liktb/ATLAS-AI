# ATLAS: Automated Tracking of Land, Agriculture & Sprawl: [Web Application](https://atlasai-109.streamlit.app/)

**Track:** AI + Climate 
**Team Name:** ATLAS-AI  
**Team Members:** Ryan Dinto, Aaryan Kalra, Devansh Sharma  

---

> **Disclaimer:** *This project is a prototype for geospatial environmental monitoring and is not official municipal planning or regulatory advice[cite: 31].*

---

##  The Problem
Rapid urban expansion and coastal land reclamation (such as artificial island megaprojects in Abu Dhabi like Ramhan Island) drastically alter fragile local ecosystems, leading to the loss of natural sabkha, coastal wetlands, and unmanaged environmental risk. Urban planners and environmental researchers require rapid, quantitative tools to measure historical land-cover changes and reliably project future environmental degradation.

##  What We Built
ATLAS is a multi-modal geospatial AI platform available as both a Streamlit web command-center interface (`app.py`) and a standalone CLI analytical engine (`run_analysis.py`). It takes historical and modern Sentinel-2 satellite imagery tiles sourced from Planet Insights, computes compositional land-cover distributions using a fine-tuned ResNet-18 neural network, and runs scientific logistic-growth mathematical models to project future urban sprawl velocity and water scarcity risk indices.

##  What We Added
While standard models output independent multi-label probabilities that do not form a coherent whole, we built a **Compositional 100% Softmax Normalization Engine** so that all land-cover classes sum strictly to 100% of the image area. Furthermore, we implemented **Arid-Zone Radiometric Domain Adaptation** to prevent false European forest classifications over Middle Eastern deserts, alongside a **Predictive Simulation Engine** modeling temporal expansion velocity.

##  How It Works
* **Data (Lead: Aaryan Kalra):** Sourced and structured high-resolution satellite imagery tiles from **Planet Insights** (`https://insights.planet.com/`), anchored by regional validation case studies on Ramhan Island, Abu Dhabi (comparing 2016 baseline sabkha against 2026 urban waterfront megaprojects).
* **Model & Backend (Lead: Ryan Dinto):** Fine-tuned a ResNet-18 architecture initialized with ImageNet weights, modified for 19 multi-label classes on the BigEarthNet v2.0 (reBEN) dataset, and accelerated locally via NVIDIA CUDA with automated CPU fallback. Evaluated to achieve an **83.28% micro mAP**.
* **Interface & App Integration (Lead: Devansh Sharma):** Built a dark-themed geospatial command center in Streamlit (`app.py`) featuring interactive swipe-comparison viewports (`streamlit-image-comparison`), macro telemetry progress bars, and probability distribution shift charts via Plotly.

##  Results & Evaluation Metrics
Evaluated on held-out benchmark splits:
- **Micro Mean Average Precision (mAP):** `83.28%`
- **Macro Mean Average Precision (mAP):** `66.53%`
- **Micro F1-Score:** `72.65%`
- **Hamming Loss (Error Rate):** `7.90%`
- **Where it fails:** High-frequency coastal sun glint and shallow water reflections can occasionally skew surface water probability bounds, requiring strict radiometric edge-texture gating to prevent minor misclassifications.

##  What We'd Do Next
1. Integrate live Copernicus/Planet API streaming to query live satellite tiles dynamically for any global GPS coordinate.
2. Expand the validation set across multiple international coastal reclamation zones to further refine arid-zone edge weights.

##  Sources & Acknowledgments
* **Satellite Imagery:** Planet Insights ([https://insights.planet.com/](https://insights.planet.com/))
* **Dataset & Architecture:** BigEarthNet v2.0 (reBEN) multi-label benchmark dataset; PyTorch & Torchvision ResNet-18.
* **Libraries:** Streamlit, Plotly, Pandas, NumPy, and `streamlit-image-comparison`.
* **AI Development Assistance:** Built with coding and architecture support from **Qwencoder3.5 (9B) LLM** and **Gemini**.
  
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
