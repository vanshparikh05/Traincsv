
Naive Bayes & K-Nearest Neighbors from Scratch

 Dataset

The project uses the Titanic dataset (`train.csv`), a classic machine learning dataset from Kaggle that contains details of passengers aboard the Titanic, along with information on whether each passenger survived or not.

---

## 🔧 Step 1: Import Libraries

The first step imports essential libraries:

* `pandas` and `numpy` for data manipulation
* `train_test_split` from `sklearn.model_selection` to divide the dataset into training and testing sets.

---

## 📥 Step 2: Data Loading & Preprocessing

1. **Reading the Data**:
   The Titanic training data is loaded using `pd.read_csv`.

2. **Dropping Irrelevant Features**:
   Columns like `PassengerId`, `Name`, `Ticket`, and `Cabin` are removed as they don’t add predictive value.

3. **Handling Missing Values**:

   * Missing ages are filled with the median age.
   * Missing embarkation points are filled with the mode.

4. **Encoding Categorical Variables**:

   * Gender (`Sex`) is converted: male → 0, female → 1
   * Embarked port (`Embarked`) is converted: S → 0, C → 1, Q → 2

5. **Feature Scaling**:

   * The dataset is standardized using z-score normalization (mean 0, std dev 1) to help distance-based algorithms like KNN perform better.

6. **Train-Test Split**:

   * The data is split into 80% training and 20% testing sets for evaluating model performance.

---

## 📊 Step 3: Naive Bayes Classifier (From Scratch)

Naive Bayes is a probabilistic classification technique based on Bayes’ Theorem. It assumes that features are independent of each other given the class.

**Implementation Highlights:**

* For each class (Survived/Not Survived), the model calculates the mean and variance of each feature.
* Class priors (probability of survival or not) are also stored.
* During prediction, it calculates the likelihood of a data point belonging to each class using the Gaussian distribution formula and chooses the class with the highest posterior probability.

**Result:**

* The accuracy is calculated by comparing predictions to the test set.
* Achieved accuracy is printed (e.g., around 77%).

---

## 📈 Step 4: K-Nearest Neighbors Classifier (From Scratch)

KNN is a simple algorithm that classifies a new data point by looking at the majority class among its `k` nearest neighbors in the training set.

Implementation Highlights:

* The Euclidean distance is calculated between a test point and every training point.
* The `k` points with the smallest distances are selected.
* The class that appears most frequently among those neighbors is predicted.

Result:

* Predictions are made for all test points and accuracy is calculated.
* Typically gives higher accuracy than Naive Bayes on this dataset (around 82%).

Final Output

* The program prints the accuracy of both models.
* Models are implemented from scratch without using libraries like `scikit-learn`.

