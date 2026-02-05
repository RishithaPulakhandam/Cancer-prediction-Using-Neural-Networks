# Cancer Gene Expression Classification using Neural Networks

## Overview

This project demonstrates a complete machine learning pipeline for **classifying cancer types using gene expression data**. A **feedforward neural network (multilayer perceptron)** is trained to predict cancer types based on high-dimensional gene expression features.

The workflow includes:

* Data loading and exploration
* Data cleaning and preprocessing
* Label encoding and normalization
* Train/validation/test split
* Neural network model building using **TensorFlow/Keras**
* Model training, evaluation, and visualization

---

## Dataset

* **Source:** Kaggle – *Cancer Gene Expression Data*
* **File:** `cancer_gene_expression.csv`
* **Features:** Gene expression values (numerical)
* **Target:** `Cancer_Type` (categorical)
* **Number of Classes:** 5 cancer types

> The last column represents the class label and is not used as a feature.

---

## Requirements

The following Python libraries are required:

* numpy
* pandas
* matplotlib
* seaborn
* scikit-learn
* tensorflow / keras

This environment is compatible with the **Kaggle Python 3 Docker image**.

---

## Project Structure

1. **Data Loading**
   The dataset is read directly from the Kaggle input directory. <br>(/kaggle/input/cancer-gene-expression-data/cancer_gene_expression.csv)

2. **Exploratory Data Analysis (EDA)**

   * Dataset shape inspection
   * Column inspection
   * Missing value checks
   * Class distribution analysis

3. **Preprocessing**

   * Feature/label separation
   * Label encoding using `LabelEncoder`
   * Train/validation/test split
   * Feature scaling using `MinMaxScaler`

4. **Model Architecture**

   * Input layer: Gene expression features
   * Hidden Layer 1: 40 neurons (ReLU)
   * Hidden Layer 2: 20 neurons (ReLU)
   * Output Layer: Softmax activation for multi-class classification

5. **Training Configuration**

   * Optimizer: Adam (learning rate = 0.001)
   * Loss Function: Sparse Categorical Crossentropy
   * Batch Size: 32
   * Epochs: 200

6. **Evaluation & Visualization**

   * Accuracy evaluation on test data
   * Prediction comparison (actual vs predicted labels)
   * Training vs validation accuracy plot
   * Training vs validation loss plot

---

## Model Performance

* Accuracy and loss are tracked for both training and validation datasets.
* Visualization helps identify overfitting or underfitting trends.
* Final evaluation is performed on a held-out test set.

---

## Results

### Prediction Results

The trained neural network demonstrates strong predictive performance on the held-out test set. A comparison of predicted versus actual class labels for the first 20 test samples shows that the majority of predictions match the true cancer classes, indicating effective learning of gene expression patterns associated with each cancer type.

Example observations:

* Most samples show **exact matches between predicted and actual labels** (e.g., classes 0, 1, 3, and 4).
* Only a small number of misclassifications are observed, such as predicting class 3 instead of class 1, which is expected in multi-class biological data where expression profiles can partially overlap.

Overall, these results suggest that the model generalizes well to unseen data and is capable of distinguishing between multiple cancer types using gene expression features.

### Model Performance Graphs

* **Model Accuracy Plot:**
   <br><img width="569" height="442" alt="image" src="https://github.com/user-attachments/assets/8f488cd1-52e1-4df5-a94f-cf7947c2c3cb" /> <br>


  * Training and validation accuracy steadily increase over epochs and remain close to each other.
  * This indicates stable learning and minimal overfitting.

* **Model Loss Plot:**
   <br><img width="585" height="447" alt="image" src="https://github.com/user-attachments/assets/2a183802-3cdf-4058-90c0-4e77eb7aa35e" /> <br>


  * Both training and validation loss decrease consistently and plateau toward later epochs.
  * The absence of divergence between training and validation loss further supports good model generalization.

Together, these curves confirm that the neural network converges effectively and maintains balanced performance across training and validation datasets.

---

## How to Run

1. Ensure the dataset is available in the Kaggle input directory.
2. Run the notebook sequentially from top to bottom.
3. Review printed outputs and plotted graphs for performance insights.

---

## Future Improvements

* Apply **cross-validation** for robustness
* Add **dropout or batch normalization** to reduce overfitting
* Experiment with **deeper architectures** or **CNNs** for gene expression data
* Perform **feature selection or dimensionality reduction (PCA)**
* Compare with traditional ML models (SVM, Random Forest)

---

## Author

**Rishitha Pulakhandam**
Bioinformatics | Machine Learning | Genomics
