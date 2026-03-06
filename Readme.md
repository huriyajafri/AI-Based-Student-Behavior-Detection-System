## **AI Based Student Behavior Detection System**

## **1\. Introduction**

This project focuses on developing an automated attentiveness estimation system using computer vision and machine learning. The goal is to analyze facial cues, gaze direction, and temporal frame sequences to classify attentiveness into three levels: low, moderate, and high. The system replaces subjective manual scoring with an automated, scalable, and objective pipeline.

The project objectives include:

• Building a complete end-to-end preprocessing and modeling pipeline.

• Extracting meaningful facial features using MediaPipe.

• Training and comparing three different models.

• Evaluating performance with standard metrics and visualizations.

## **2\. Methodology**

**2.1 Data Pipeline and Preprocessing**

The dataset consisted of classroom videos captured at 10 FPS. A robust preprocessing pipeline was developed to handle the raw data:

* The dataset consists of multiple subjects, video frames, and JSON label files where each timestamp corresponds to an attentiveness score (1–5). To reduce class imbalance, labels were mapped into three classes:  
*  Low Attentive (Class 0): Label 1 and 2  
*  Moderate Attentive (Class 1): Label 3  
*  High Attentive (Class 2): Labels 4 and 5

* **Data Cleaning (v1 to v2):** Initial experiments (v1) used a raw dataset with severe class imbalance (58% Mid, 13% High). The refined dataset (v2) utilized stratified sampling and augmentation to achieve a healthier distribution: Low (\~30%), Mid (\~47%), and High (\~23%).  
* **Face Detection & Stabilization:** A "Largest Face" logic was implemented using MediaPipe to filter out background students, ensuring the model focused only on the primary subject.  
* **Lazy Loading Strategy:** To handle the 12GB+ dataset on limited RAM, a custom LazyVideoDataset class was built. This employed memory mapping (mmap\_mode=r) to load specific video clips from the disk only when required by the training batch, preventing memory overflows.

### **2.2 Model Architectures**

Three distinct models were trained and evaluated:

#### **Model A: Baseline (CNN-LSTM)**

* **Architecture:** A simple custom CNN for spatial feature extraction followed by a single LSTM layer for temporal modeling.  
* **Input:** Sequences of 16 frames (224 x  224 pixels).  
* **Loss Function:** Standard Cross-Entropy Loss.  
* **Purpose:** To establish a performance baseline and identify dataset limitations.

#### **Model B: Refined Hybrid (ResNet18 \+ LSTM)**

* **Architecture:** A **ResNet-18** backbone (pre-trained on ImageNet) replaced the custom CNN to extract robust spatial features. The last two layers were unfrozen for fine-tuning. A standard LSTM processed the sequence of features.  
* **Optimization:** Utilized **AdamW** optimizer with weight decay and a **ReduceLROnPlateau** scheduler to dynamically adjust the learning rate.  
* **Class Balancing:** Applied calculated **Class Weights** (Wlow\=1.13, Wmid\=0.71, Whigh\=1.42) to the loss function to penalize misclassifications of the minority "High Attention" class.

#### **Model C: Advanced 3D-CNN (R(2+1)D)**

* **Architecture:** A **ResNet (2+1)D** model pre-trained on the Kinetics-400 video dataset. This model explicitly learns spatiotemporal features by performing 2D spatial convolutions followed by 1D temporal convolutions.  
* **Loss Function:** **Focal Loss**  was implemented to force the model to focus on "hard" examples (distinguishing Mid vs. High) rather than the easy "Low" class.  
* **Constraint:** Due to high VRAM usage, training required a small batch size (4) with Gradient Accumulation to simulate stable learning.

## **3\. Results**

### **3.1 Quantitative Comparison**

The models were evaluated on a held-out validation set. The performance metrics are summarized below:

| Metric | Model A (Baseline) | Model B (Refined) | Model C (R(2+1)D) |
| :---- | :---- | :---- | :---- |
| **Accuracy** | 52.0% | **81.2%** | 76% |
| **Precision (Macro)** | 0.48 | 0.80 | 0.78 |
| **Recall (Macro)** | 0.48 | 0.80 | 0.76 |
| **Training Time** | \~1 min/epoch | \~1.2 min/epoch | \~3.5 min/epoch |
| **Stability** | Low (Fluctuating) | **High (Steady)** | Low (Volatile) |

### 

### **3.2 Visual Analysis of Training Dynamics**

![][image1]

* **Model A:** The training curve showed significant fluctuation, plateauing around 52%. The confusion matrix revealed the model was collapsing into the majority class ("Mid Attention"), failing to distinguish "High Attention" entirely.  
* **Model B (Winner):** The training curve was smooth and monotonic, indicating stable convergence. The validation accuracy climbed steadily from 65% to 81.2%, proving the effectiveness of the pre-trained ResNet backbone and class weights.  
* **Model C:** While it achieved the same peak accuracy (81.2%), the loss curve was "saw-toothed" and volatile. The model required significantly more epochs to stabilize, likely due to the difficulty of training 3D parameters on a smaller dataset.

### **3.3 Prediction Analysis**

In visual inference tests (demo videos):

* **Low Attention:** All models performed well, easily identifying head-down postures (e.g., phone usage).  
* **Mid vs. High:** Model A failed completely. Model B and C successfully distinguished "Passive Listening" (Mid) from "Active Focus" (High) by detecting subtle cues like head tilt and gaze stability. Model B was notably faster at generating predictions (45 FPS vs 12 FPS for Model C).

## **4\. Discussion**

### **4.1 The Impact of Class Imbalance**

The failure of Model A highlights the critical challenge of class imbalance in student engagement datasets. Without intervention, the model optimized for the majority "Mid" class. The implementation of **Class Weights** in Model B and **Focal Loss** in Model C successfully mitigated this, raising the recall for the "High Attention" class from \~20% to over 80%.

### **4.2 Architecture Trade-offs: 2D vs. 3D**

This project compared "Late Fusion" (CNN-LSTM) against "Early Fusion" (3D-CNN).

* **CNN-LSTM (Model B)** proved superior for this specific task. Engagement signals (head pose, gaze) are largely spatial features that change slowly over time. The explicit temporal modeling of the LSTM was sufficient and computationally efficient.  
* **R(2+1)D (Model C)**, while theoretically more powerful for complex actions (like "jumping"), was overkill for engagement detection. Its high computational cost and training instability make it less suitable for real-time deployment on standard hardware.

### **4.3 Challenges Overcome**

* **Background Noise:** Initial models were confused by students sitting in the background. The "Largest Face" heuristic solved this by dynamically cropping the primary subject.  
* **Hardware Constraints:** Training 3D models on Colab's free tier caused OOM errors. This was solved by implementing Gradient Accumulation and mixed-precision training (AMP).

## 

## **5\. Conclusion**

### **5.1 Key Findings**

1. **Refined Hybrid Models are Optimal:** The **ResNet18 \+ LSTM (Model B)** offered the best balance of performance (81.2% accuracy) and efficiency. It is the recommended model for deployment.  
2. **Data Balance is Crucial:** Algorithmic improvements (like 3D convolution) cannot compensate for poor data distribution. Stratified sampling and augmentation were the primary drivers of the accuracy jump from 52% to 81%.  
3. **Real-Time Capability:** Model B operates at \>45 FPS, making it viable for live classroom monitoring systems.

### **5.2 Limitations**

* **Dataset Diversity:** The model was trained on a specific set of subjects. It may struggle with extreme lighting conditions or side profiles not present in the training set.  
* **Proxy Limitations:** The model relies heavily on head pose. It may classify a student sleeping with their head upright as "Engaged."

### **5.3 Future Work**

* **Multimodal Analysis:** Integrating audio (speech activity) or physiological sensors (heart rate) could improve robustness.  
* **Attention Mechanisms:** Replacing the LSTM with a Transformer (Self-Attention) could allow the model to better weigh "key moments" of disengagement over long video sequences.

## **6\. References**

* **\[1.1\]** "Student Engagement Detection in Classrooms through Computer Vision... Using YOLOv4", *ResearchGate*, 2025\.  
* **\[1.2\]** "Student's Engagement Detection Based on Computer Vision: A Systematic Literature Review", *IEEE Xplore*, 2025\.  
* **\[2.1\]** "Automatic Sports Video Classification Using CNN-LSTM Approach", *CEUR-WS.org*.  
* **\[3.3\]** "R(2+1)D: A Closer Look at Spatiotemporal Convolutions for Action Recognition", *GitHub*.  
* **\[4.1\]** "Computational Strategies for Handling Imbalanced Data in Machine Learning", *ISI-Web*.  
* **\[4.5\]** "Addressing Class Imbalances in Video Time-Series Data...", *MDPI*, 2024\.
