# Osteoarthritis Prediction System

An AI-powered system that predicts the likelihood of Osteoarthritis (OA) from X-ray images using multiple deep learning models.  
The project is designed to help doctors and patients with early detection and decision-making by providing accurate predictions from multiple models.

---

## Features
- Upload X-ray images to predict Osteoarthritis likelihood.
- Provides a **percentage probability score** for OA.
- Utilizes **8 trained deep learning models**:
  - DINO, InceptionV3, CoAtNet, and EfficientNet (Original & Distributed versions).
- Shows results in a table with each model’s prediction (Original vs. Distributed) for easy comparison.

---

## Dataset
The dataset used is from Kaggle: [Osteoarthritis Prediction Dataset](https://www.kaggle.com/datasets/farjanakabirsamanta/osteoarthritis-prediction)

We worked with **two versions of the dataset**:
1. **Original (OG)**  
   - Directly downloaded from Kaggle without modifications.
   - Models trained on this dataset reflect the natural class distribution.

2. **Distributed (dist)**  
   - All normal (non-OA) and OA-positive images were **merged and rebalanced** with a different Normal:OA ratio.
   - Separate models were trained on this dataset to address class imbalance and improve generalization.

---

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python
- **Deep Learning Frameworks**: PyTorch, TensorFlow/Keras

### Models Used
- DINO (Self-Supervised Vision Transformer)  
  - DINO_OG.pth, DINO_dist.pth
- InceptionV3 (CNN)  
  - IV3_OG.keras, IV3_dist.keras
- CoAtNet (Hybrid CNN + Transformer)  
  - CoAtNet_OG.keras, CoAtNet_dist.keras
- EfficientNet (CNN)  
  - EfficientNet_OG.keras, EfficientNet_dist.keras

---

## Screenshots

<p align="center">
  <img src="https://github.com/user-attachments/assets/b89196eb-331d-44c3-a93d-d03822098bf6" 
       alt="Screenshot 1" width="80%" />
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/4258e7d9-d19d-432c-920d-ee934111723c" 
       alt="Screenshot 2" width="80%" />
</p>

