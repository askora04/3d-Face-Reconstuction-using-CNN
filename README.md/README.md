# 🧑‍💻 Skull2Face: 3D Face Reconstruction from Skulls using AI

This project explores **3D face reconstruction** from skull datasets and photos using **deep learning, parametric models (FLAME/3DMM), and AI-powered texture synthesis**.  
It provides a **complete pipeline**: from preprocessing skull meshes → predicting 3D face geometry → generating realistic skin textures with Blender plugins (Stable Diffusion integration).

---

## 📌 Features
- Preprocess skull meshes (CT → Mesh, OBJ/STL → Clean/Normalize).
- Landmark extraction and tissue depth priors.
- Skull → FLAME parameter regression using a **CNN/PointNet encoder**.
- Geometry reconstruction with **FLAME** (parametric face model).
- Texture generation with **Stable Diffusion** / Blender add-ons.
- Training, inference, and visualization scripts included.
- Modular repo structure for easy extension.

---

## 🔬 Workflow

```
   Skull Mesh / CT → Preprocessing → Skull Encoder (CNN/PointNet)
            ↓                                ↓
        Normalized mesh         →     FLAME Param Predictor
                                        ↓
                        3D Face Mesh (geometry)
                                        ↓
                        Blender + AI Texturing
                                        ↓
                          Realistic Face Model
```

---

## 📂 Project Structure
```
skull2face/
├── data/                 # skulls + optional ground-truth faces
├── tools/                # mesh I/O, photogrammetry, DICOM→mesh
├── preprocess/           # alignment, normalization, landmarks
├── models/               # FLAME wrapper + Skull→Face regressor
├── training/             # train loops, losses
├── inference/            # predict face from skull
├── blender_addon/        # plugin for AI-driven texture generation
├── notebooks/            # experiments & visualizations
└── README.md             # this file
```

---

## ⚙️ Installation

```bash
# Create environment
conda create -n skull2face python=3.9 -y
conda activate skull2face

# Install dependencies
pip install torch open3d trimesh pytorch3d numpy scipy imageio

# (Optional) Install photogrammetry tools
# openMVG / openMVS must be installed separately

# For Blender texture synthesis:
# Install Blender (>=3.5) and add plugins (StableGen / Texture Diffusion)
```

---

## 🚀 Usage

### 1. Preprocess skull mesh
```bash
python tools/mesh_utils.py --input data/skulls/skull01.obj --output data/skulls/normalized_skull01.ply
```

### 2. Train model
```bash
python training/train.py --config configs/train.yaml
```

### 3. Inference (reconstruct face)
```bash
python inference/infer.py --skull data/skulls/normalized_skull01.ply --out out/face.obj
```

### 4. Texture generation in Blender
- Open Blender → Import `out/face.obj`.  
- Run **Generate Skin Texture (AI)** operator from the provided plugin.  
- A realistic skin texture will be synthesized and applied to the 3D mesh.

---

## 📊 Datasets
- **Skull-face paired datasets** (e.g., Skull100) are required for supervised training.  
- If unavailable, synthetic skulls can be generated from head scans by erosion.  
- For textures, use large-scale face datasets (CelebA-HQ, BU-3DFE, etc.).  

> ⚠️ Many datasets are restricted (medical/forensic). Make sure you have proper **licenses and permissions** before use.

---

## 📜 Disclaimer
This project is for **research and educational purposes only**.  
Reconstructing faces from skulls has **ethical, legal, and privacy implications**.  
Do not apply this code to real forensic or law enforcement cases without proper authorization.

---

## 🙌 Acknowledgements
- [FLAME: Learning a model of facial shape & expression](https://flame.is.tue.mpg.de/)  
- [OpenMVG & OpenMVS](https://github.com/openMVG/openMVG)  
- [Stable Diffusion & Blender Add-ons (Texture Diffusion, StableGen)]  

---

## 💡 Future Work
- Integrate implicit neural representations (NeRF/DeepSDF).  
- Add human-in-the-loop refinement (interactive Blender UI).  
- Improve tissue-thickness prior conditioning.  

---

## 🧑 Author
Maintained by **[Your Name]** ✨  
Feel free to fork, star ⭐, and contribute via pull requests!
