# IEEE-CIS Fraud Detection (თაღლითური ტრანზაქციების აღმოჩენა)

## პროექტის მიმოხილვა:

მოცემული პროექტი ეფუძნება Kaggle-ის პლატფორმაზე არსებულ კონკურსს, რომელიც საბანკო ტრანზაქციების კლასიფიკაციას ისახავს მიზნად, კერძოდ თაღლითურია (isFraud=1) ის, თუ ლეგიტიმური (isFraud=0). მონაცემები მოიცავს 590,000-ზე მეტ ტრანზაქციას ორი ცხრილის სახით და შეფასება ხდება ROC-AUC მეტრიკით. ერთ-ერთი მთავარი გამოწვევა ისაა, რომ მონაცემები ძალზედ დაუბალანსებელია, მხოლოდ ~3.5% ტრანზაქცია არის თაღლითური.

---

## ჩემი მიდგომა პრობლემის გადასაჭრელად:

1. **მონაცემთა გასუფთავება (Cleaning):** გამოვიყენე სამ ეტაპიანი სტრატეგია: მაღალი missing-rate სვეტების წაშლა, numeric NaN-ების შევსება მედიანით და categorical NaN-ების შევსება mode-ით.
2. **Feature Engineering:** ახალი სიგნალების შექმნა, რომლებიც fraud-ის ნიმუშებს უკეთ ასახავს: log-transformed amount, decimal feature, hour და day features, label encoding.
3. **Feature Selection:** 3 მეთოდის თანმიმდევრული გამოყენება: Variance Threshold, Correlation Filter და Random Forest Importance.
4. **მოდელირება და ანალიზი:** 7 სხვადასხვა არქიტექტურის ტესტირება (ინდივიდუალურ notebook-ში), hyperparameter-ების ვარიაციით Underfitting-ისა და Overfitting-ის ანალიზი, 5-Fold Cross-Validation სტაბილურობის შემოწმებისთვის.
5. **Pipeline:** თითოეული მოდელი შენახულია Pipeline-ად, რომელიც პირდაპირ raw test data-ზე მუშაობს.

---

## რეპოზიტორიის სტრუქტურა

```
model_experiment_XGBoost.ipynb            XGBoost ექსპერიმენტები
model_experiment_RandomForest.ipynb       Random Forest ექსპერიმენტები
model_experiment_LogisticRegression.ipynb Logistic Regression ექსპერიმენტები
model_experiment_DecisionTree.ipynb       Decision Tree ექსპერიმენტები
model_experiment_GradientBoosting.ipynb   Gradient Boosting ექსპერიმენტები
model_experiment_AdaBoost.ipynb           AdaBoost ექსპერიმენტები
model_experiment_LinearModels.ipynb       Ridge Classifier ექსპერიმენტები
model_inference.ipynb                     საბოლოო submission გენერაცია
```

## ფაილების შინაარსი

**`train_transaction.csv / train_identity.csv:`** სასწავლო მონაცემები. დამუშავება: merge-ის შემდეგ 590,540 ტრანზაქცია, 434 სვეტი. target სვეტია `isFraud`.

**`test_transaction.csv / test_identity.csv:`** სატესტო მონაცემები. დამუშავება: merge-ის შემდეგ 506,691 ტრანზაქცია (`isFraud`-ის გარეშე). გამოიყენება საბოლოო submission-ისთვის.

**`model_experiment_{arch}.ipynb:`** თითოეული არქიტექტურისთვის ცალკე notebook. შეიცავს Cleaning, Feature Engineering, Feature Selection და Training სექციებს, MLflow logging-ს და Pipeline რეგისტრაციას.

**`model_inference.ipynb:`** ჩატვირთავს MLflow Model Registry-დან champion მოდელს და დააგენერირებს submission.csv-ს.

---

## Feature Engineering

cleaning-ის შემდეგ მნიშვნელოვანი ნაბიჯი იყო ახალი feature-ების შექმნა, რომლებიც fraud-ის სიგნალებს ასახავს, raw სვეტებში ეს ინფორმაცია პირდაპირ არ ჩანს, ამიტომ დამჭირდა გარდაქმნა.

### 1. Cleaning მიდგომები

cleaning-ის ეტაპზე სამი ძირითადი მიდგომა გამოვიყენე:
- **50% Threshold:** 50%-ზე მეტი missing value-ის სვეტები მთლიანად ამოვშალე
- **Numeric → Median:** რიცხვითი სვეტების NaN-ები median-ით შევავსე
- **Categorical → Mode:** კატეგორიული სვეტების NaN-ები mode-ით შევავსე

### 2. NaN მნიშვნელობების შევსება (Imputation)

ამ dataset-ში NaN-ების დამუშავება 3 ეტაპად განხორციელდა:

- **50% Threshold:** სვეტები, სადაც მნიშვნელობების ნახევარზე მეტი დაკლებულია, საერთოდ წაიშლება. ასეთი სვეტების imputation შეცდომებს გამოიწვევდა, ვინაიდან არარსებული ინფორმაციის "გამოგონება" მოდელს შეაცდენდა.

- **Numeric NaN → Median:** რიცხვითი სვეტებისთვის გამოვიყენე median და არა mean, ვინაიდან ტრანზაქციების თანხები მძიმედ right-skewed-ია: ერთი დიდი ტრანზაქცია mean-ს ძლიერ გადაწევდა, median კი ამ პრობლემისადმი მდგრადია.

- **Categorical NaN → Mode:** კარტის ტიპის, ელ-ფოსტის დომენისა და სხვა კატეგორიული სვეტებისთვის ყველაზე ხშირი მნიშვნელობა გამოვიყენე default-ად.

### 3. კატეგორიული ცვლადების კოდირება

- **Label Encoding:** ყველა კატეგორიული სვეტისთვის გამოვიყენე Label Encoding. ხეებზე დაფუძნებული მოდელები კარგად ახერხებენ ordinal integer-ების დამუშავებას. One-Hot Encoding 300+ კატეგორიულ სვეტზე feature space-ს გადაჭარბებულად გაზრდიდა.

### 4. ახალი Feature-ების შექმნა

- **TransactionAmt_log:** transaction amount-ის log1p ტრანსფორმაცია, რაც ანეიტრალებს right skew-ს.
- **TransactionAmt_decimal:** თანხის ცენტობრივი ნაწილი: fraudster-ები ხშირად იყენებენ თანხებს როგორიცაა 99.00 ან 0.01.
- **Transaction_hour:** TransactionDT-დან საათი: fraud ღამის საათებში უფრო ხშირია.
- **Transaction_day:** TransactionDT-დან კვირის დღე: fraud-ს ხშირად აქვს ტემპორალური ნიმუში.

---

## Feature Selection

Cleaning-ისა და feature engineering-ის შემდეგ 222 სვეტი დარჩა. ბევრი სვეტი ნიშნავს მეტ ინფორმაციას, მაგრამ ზედმეტი "ხმაური" (noise) მოდელს აბნევს და Overfitting-ის რისკს ზრდის, ამიტომ სამი კონკრეტული მეთოდი გამოვიყენე თანმიმდევრულად:

### 1. Variance Threshold (threshold=0.01)
თითქმის მუდმივი მნიშვნელობის სვეტები ვერ ეხმარება მოდელს გადაწყვეტილების მიღებაში, ამიტომ ამოვშალე ისინი.

### 2. Correlation Filter (threshold=0.95)
0.95-ზე მაღალი კორელაციის წყვილი feature-ები ერთმანეთთან redundant ინფორმაციას შეიცავს, ამიტომ ერთ-ერთი ამოვშალე.

### 3. Random Forest Importance (Top 50)
მსუბუქი Random Forest (50 ხე, depth=8) ვატრენინგე და feature-ები importance-ის მიხედვით დავალაგე. ტოპ 50 დავტოვე.

**დასკვნა:** 222 სვეტიდან 50-მდე შემცირება მოდელის სიჩქარეს და სტაბილურობას მნიშვნელოვნად აუმჯობესებს, საჭირო ინფორმაციის დაკარგვის გარეშე.

---

## Training

### მოდელების წვრთნა და შეფასება (Training & Evaluation)

ყველა მოდელი დავყავი 80/20 split-ით და 5-Fold Stratified Cross-Validation-ით. Stratified გამოვიყენე იმიტომ, რომ fraud-ის პროპორცია (3.5%) ორივე set-ში ერთნაირად შენარჩუნებულიყო. ყველა run MLflow-ში ავირჩიე სპეციალური overfit_gap მეტრიკით (train_auc - val_auc).

---

### 1. Ridge Classifier (Linear Model)

Ridge გამოვიყენე წრფივი baseline-ის სახით. alpha პარამეტრის ვარირებით (0.01-დან 1000-მდე) დავაკვირდი underfitting-ისა და overfitting-ის ეფექტს.

| Alpha | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 0.01 | 0.79 | 0.78 | 0.00 |
| 0.1 | 0.79 | 0.78 | 0.00 |
| 1.0 | 0.79 | 0.78 | 0.00 |
| 10.0 | 0.79 | 0.78 | 0.00 |
| 100.0 | 0.79 | 0.78 | 0.00 |
| 1000.0 | 0.79 | 0.78 | 0.00 |

**დასკვნა:** val_auc 0.78-ზე გაიჭედა ყველა alpha-სთვის. ეს არქიტექტურული underfitting-ია, წრფივი მოდელი ვერ ახერხებს fraud-ის კომპლექსური არახაზოვანი ნიმუშების ათვისებას, alpha-ს ცვლილება ვერაფერს უშველის.

![Ridge Results](Linear_Regression.png)

---

### 2. Logistic Regression

C პარამეტრის ვარირებით (0.001-დან 100-მდე) გავტესტე regularization-ის ეფექტი.

| C | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 0.001 | 0.72 | 0.72 | 0.00 |
| 0.01 | 0.72 | 0.72 | 0.00 |
| 0.1 | 0.72 | 0.72 | 0.00 |
| 1.0 | 0.72 | 0.72 | 0.00 |
| 10.0 | 0.72 | 0.72 | 0.00 |
| 100.0 | 0.72 | 0.72 | 0.00 |

**დასკვნა:** val_auc 0.72-ზე გაიჭედა ყველა C-სთვის: Ridge-ის მსგავსი სიტუაცია. Logistic Regression-ს ამ ამოცანაზე არ შეუძლია fraud-ის ნიმუშების სრულყოფილი ათვისება. ეს underfitting-ის კლასიკური მაგალითია, model-ი არქიტექტურულად ზედმეტად მარტივია.

![Logistic Regression Results](Logistic_Regression.png)

---

### 3. Decision Tree — Overfitting/Underfitting-ის დემონსტრაცია

Decision Tree ყველაზე ნათლად ასახავს bias-variance trade-off-ს. max_depth-ის ზრდასთან ერთად ნათლად ჩანს გადასვლა underfitting-დან overfitting-ზე.

| max_depth | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 2 | 0.50 | 0.50 | 0.00 |
| 3 | 0.68 | 0.67 | 0.01 |
| 5 | 0.77 | 0.72 | 0.05 |
| 8 | 0.86 | 0.83 | 0.03 |
| 10 | 0.89 | 0.83 | 0.06 |
| 12 | 0.84 | 0.81 | 0.03 |
| 15 | 0.82 | 0.83 | -0.01 |
| None | 1.00 | 0.83 | **0.23** |

**Underfitting:** depth=2-3 ხე ძალიან მარტივია, val_auc=0.50-0.67, ვერ ათვისებს fraud-ის ნიმუშებს.

**Overfitting:** depth=None train_auc=1.00, val_auc=0.83, gap=0.23. ხე training set-ს მთლიანად ზეპირად ისწავლის, ახალ მონაცემებზე კი ხშირად შეცდება.

![Decision Tree Results](Decision_Tree.png)

---

### 4. Random Forest: Overfitting-ის Mitigation

n_estimators-ისა და max_depth-ის ვარირებით გავარკვიე, სად იწყება overfitting.

#### n_estimators sweep:

| n_estimators | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 50 | 1.00 | 0.91 | 0.09 |
| 100 | 1.00 | 0.92 | 0.08 |
| 200 | 1.00 | 0.92 | 0.08 |

#### max_depth sweep:

| max_depth | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 3 | 0.78 | 0.76 | 0.02 |
| 5 | 0.84 | 0.82 | 0.02 |
| 10 | 0.95 | 0.90 | 0.05 |
| 20 | 0.99 | 0.91 | 0.08 |
| None | 1.00 | 0.92 | 0.08 |

**დასკვნა:** Random Forest train_auc-ზე 1.00-ს აღწევს ყველა კონფიგურაციაში, val_auc კი 0.92-ს, ეს overfitting-ის ნიშანია. რამდენიმე ხის გაერთიანებამ Decision Tree-ს შედეგი გააუმჯობესა, მაგრამ overfitting მთლიანად ვერ აღმოფხვრა.

![Random Forest Results](Random_Forest.png)

---

### 5. GradientBoosting: Sequential Error Correction

n_estimators-ის sweep გამოვიყენე underfitting-იდან სტაბილურ მდგომარეობაში გადასვლის სადემონსტრაციოდ.

| n_estimators | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 50 | 0.86 | 0.86 | 0.00 |
| 100 | 0.87 | 0.86 | 0.00 |
| 200 | 0.88 | 0.87 | 0.00 |

**დასკვნა:** GradientBoosting-მა ძალიან კარგი ბალანსი აჩვენა, overfit gap თითქმის ნულია ყველა კონფიგურაციაში, თუმცა val_auc 0.87-ს ვერ სცილდება, რაც XGBoost-ს ჩამოუვარდება.

![GradientBoosting Results](Gradient_Boosting.png)

---

### 6. AdaBoost: სუსტი Ensemble ამ ამოცანაზე

n_estimators-ისა და learning_rate-ის ვარირებით გავტესტე AdaBoost-ის ქცევა.

| n_estimators | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 10 | 0.55 | 0.55 | 0.00 |
| 50 | 0.84 | 0.84 | 0.00 |
| 100 | 0.84 | 0.84 | 0.00 |
| 150 | 0.85 | 0.85 | 0.00 |
| 200 | 0.85 | 0.84 | 0.00 |

**დასკვნა:** AdaBoost n=10-ზე აშკარა underfitting-ს აჩვენებს (val_auc=0.55), შემდეგ სტაბილურდება 0.84-0.85-ზე. Overfit gap ნულთან ახლოსაა, მაგრამ val_auc XGBoost-ს მნიშვნელოვნად ჩამოუვარდება. AdaBoost imbalanced dataset-ებზე ყველაზე სუსტი ensemble მეთოდია, რადგან outlier-ებისადმი მგრძნობიარეა.

![AdaBoost Results](plots/AdaBoost.png)

---

### 7. XGBoost — Champion Model

learning_rate-ისა და max_depth-ის sweep-ებმა ნათლად გვიჩვენა ოპტიმალური კონფიგურაცია.

#### Learning Rate Sweep:

| learning_rate | n_estimators | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|---|
| 0.3 | 50 | 0.88 | 0.87 | 0.01 |
| 0.1 | 500 | 0.96 | 0.95 | 0.01 |
| 0.05 | 500 | 0.95 | 0.94 | 0.01 |
| 0.01 | 1000 | 0.94 | 0.93 | 0.01 |

#### Depth Sweep:

| max_depth | Train AUC | Val AUC | Overfit Gap |
|---|---|---|---|
| 2 | 0.88 | 0.87 | 0.01 |
| 3 | 0.89 | 0.87 | 0.02 |
| 6 | 0.98 | 0.95 | 0.03 |
| 9 | 0.94 | 0.93 | 0.01 |
| 12 | 0.99 | 0.96 | 0.03 |

#### Best Config (Regularized):

| config | Train AUC | Val AUC | CV AUC | Overfit Gap |
|---|---|---|---|---|
| XGBoost_regularized_best | 0.99 | **0.96** | **0.95** | 0.03 |

**დასკვნა:** depth=2-3 underfitting-ს იძლევა (0.87), depth=12 overfitting-ს იწყებს (gap=0.04), depth=6 ოპტიმალურია, regularization (subsample=0.8, reg_alpha=0.1) კი overfit gap-ს ამცირებს.

![XGBoost Results](XGBoost.png)

---

### Hyperparameter ოპტიმიზაციის მიდგომა

ამ პროექტში გამოვიყენე **manual grid search**. ეს უფრო ნელია, მაგრამ:

1. ჩანს **ყოველი გადაწყვეტილების შედეგი**.
2. ჩანს underfitting/overfitting pattern-ი ყოველი model-ისთვის ცალ-ცალკე.
3. MLflow-ში **ყველა run ლოგირდება**.

Grid-ი:
- `Ridge`: `alpha ∈ {0.01, 0.1, 1.0, 10.0, 100.0, 1000.0}`
- `LogisticRegression`: `C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}`
- `DecisionTree`: `max_depth ∈ {2, 3, 5, 8, 10, 12, 15, None}`
- `RandomForest`: `n_estimators ∈ {50, 100, 200}`, `max_depth ∈ {3, 5, 10, 20, None}`
- `GradientBoosting`: `n_estimators ∈ {50, 100, 200}`
- `AdaBoost`: `n_estimators ∈ {10, 50, 100, 150, 200}`, `learning_rate ∈ {0.1, 0.5, 1.0, 2.0}`
- `XGBoost`: `learning_rate ∈ {0.3, 0.1, 0.05, 0.01}`, `max_depth ∈ {2, 3, 6, 9, 12}`

---

### საბოლოო Model-ის შერჩევის დასაბუთება

**XGBoost (regularized_best)** შეირჩა, რადგან:

1. ყველაზე მაღალი Val AUC: **0.96** (სხვა მოდელებს მნიშვნელოვნად სჯობს)
2. CV AUC = **0.95**: სტაბილური შედეგი cross-validation-ზეც
3. Overfit Gap = **0.03**: მინიმალური სხვაობა train-val AUC-ს შორის
4. L1/L2 regularization (reg_alpha=0.1, reg_lambda=1.0): აკონტროლებს სირთულეს
5. Subsampling (subsample=0.8, colsample_bytree=0.8): ამატებს randomness-ს
6. Ridge/LogReg-ს სჯობს, რადგან non-linear კავშირებს იჭერს
7. Decision Tree-ს სჯობს, რადგან overfitting-ს resistance აქვს
8. AdaBoost-ს სჯობს, რადგან imbalanced data-ზე robust-ია

---

### Model Comparison — საბოლოო შედარება

| მოდელი | Best Val AUC | Train AUC | Overfit Gap | შეფასება |
|---|---|---|---|---|
| **XGBoost** | **0.96** | 0.99 | 0.03 | **Champion** |
| RandomForest | 0.92 | 1.00 | 0.08 | კარგი, overfitting |
| GradientBoosting | 0.87 | 0.88 | 0.00 | კარგი ბალანსი |
| AdaBoost | 0.85 | 0.85 | 0.00 | სუსტი imbalanced data-ზე |
| DecisionTree | 0.83 | 1.00 | 0.23 | Severe overfitting |
| Ridge | 0.78 | 0.79 | 0.00 | Underfitting |
| LogisticRegression | 0.72 | 0.72 | 0.00 | Underfitting |

---

## MLflow Tracking

იმის ნაცვლად, რომ სხვადასხვა hyperparameter-ით მიღებული შედეგები ხელით ჩაგვეწერა, გამოვიყენე MLflow. მისი მეშვეობით, ყველა run ავტომატურად იგზავნება DagsHub-ის პლატფორმაზე. ეს საშუალებას გვაძლევს, ერთიან სივრცეში შევადაროთ ყველა ვარიანტი და მარტივად დავინახოთ, პარამეტრების რომელი კომბინაცია იძლევა საუკეთესო შედეგს.

ამ დავალების სპეციფიკაც სწორედ ისაა, რომ **ყველა მოდელის არქიტექტურისთვის ცალკე ექსპერიმენტი შეგვექმნა** MLflow-ზე. თითოეული ექსპერიმენტის შიგნით კი ცალ-ცალკე run-ები შეესაბამება cleaning-ს, feature selection-ს, training-ის თითოეულ კონფიგურაციასა და pipeline რეგისტრაციას.

## MLflow ექსპერიმენტები:

- [DagsHub Experiments](https://dagshub.com/GigiSichinava/ML-Assignment-2.mlflow/#/experiments)

### ჩაწერილი მეტრიკების აღწერა

- **train_auc:** ROC-AUC სასწავლო სეტზე: გვიჩვენებს რამდენად კარგად ისწავლა მოდელმა.

- **val_auc:** ROC-AUC ვალიდაციის სეტზე: ეს ყველაზე მნიშვნელოვანი მეტრიკაა, რადგან გვიჩვენებს რეალურ performance-ს ახალ მონაცემებზე.

- **cv_auc_mean:** 5-Fold Cross-Validation-ის საშუალო AUC: გვიჩვენებს შედეგის სტაბილურობას სხვადასხვა split-ზე.

- **cv_auc_std:** Cross-Validation-ის სტანდარტული გადახრა: დაბალი std ნიშნავს სტაბილურ მოდელს.

- **overfit_gap:** train_auc - val_auc: ჩვენ მიერ დამატებული მეტრიკა, რომელიც პირდაპირ ზომავს overfitting-ის სიძლიერეს. 0.05-ზე მაღალი gap overfitting-ის ნიშანია.

### საუკეთესო მოდელის შედეგები

- **Model:** `XGBoost`
- **Config:** `n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, reg_alpha=0.1`
- **Val AUC:** `0.96`
- **CV AUC Mean:** `0.95`
- **Overfit Gap:** `0.03`
- **Kaggle Public Score:** `0.9137`
- **Kaggle Private Score:** `0.8878`

---

## ბმულები

- **DagsHub რეპოზიტორია:** [GigiSichinava/ML-Assignment-2](https://dagshub.com/GigiSichinava/ML-Assignment-2)
- **MLflow ექსპერიმენტები:** [DagsHub Experiments](https://dagshub.com/GigiSichinava/ML-Assignment-2.mlflow)
- **Kaggle კონკურსი:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
