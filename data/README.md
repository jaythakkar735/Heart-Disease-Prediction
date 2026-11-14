# Dataset Documentation: Heart Disease (Cleveland Clinic Foundation)

## 1. Overview & Attribution
- **Dataset Name**: Heart Disease (Cleveland Database)
- **Primary Source**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/45/heart+disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Direct Raw Data**: [https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
- **Original Creators**:
  - Andras Janosi, M.D. (Gottsegen Hungarian Institute of Cardiology, Budapest)
  - William Steinbrunn, M.D. (University Hospital, Zurich, Switzerland)
  - Matthias Pfisterer, M.D. (University Hospital, Basel, Switzerland)
  - Robert Detrano, M.D., Ph.D. (Cleveland Clinic Foundation)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

## 2. Dataset Schema (14 Attributes)
The benchmark processed Cleveland subset contains 303 patient instances and 14 clinical attributes:

| Column | Type | Description | Values / Units |
| :--- | :--- | :--- | :--- |
| `age` | Numeric | Age of patient | Years (29 - 77) |
| `sex` | Categorical | Biological sex | 1 = male, 0 = female |
| `cp` | Categorical | Chest pain type | 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic |
| `trestbps` | Numeric | Resting blood pressure | mm Hg upon hospital admission (94 - 200) |
| `chol` | Numeric | Serum cholesterol | mg/dl (126 - 564) |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| `restecg` | Categorical | Resting electrocardiographic results | 0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved during exercise | bpm (71 - 202) |
| `exang` | Categorical | Exercise induced angina | 1 = yes, 0 = no |
| `oldpeak` | Numeric | ST depression induced by exercise relative to rest | Continuous (0.0 - 6.2) |
| `slope` | Categorical | Slope of the peak exercise ST segment | 1: upsloping, 2: flat, 3: downsloping |
| `ca` | Numeric/Cat | Number of major vessels (0-3) colored by fluoroscopy | 0, 1, 2, 3 (4 missing entries as '?') |
| `thal` | Categorical | Thallium scintigraphy stress test | 3 = normal, 6 = fixed defect, 7 = reversible defect (2 missing entries as '?') |
| `target` | Integer | Angiographic disease presence | 0: <50% diameter narrowing (absence), 1-4: >50% narrowing (presence) |

## 3. Preprocessing Notes & Missing Values
- In binary classification benchmarks, `target` is transformed: `0` denotes absence of angiographic heart disease, and values `1, 2, 3, 4` are mapped to `1` (presence of coronary artery disease).
- `ca` contains 4 missing records (`?`).
- `thal` contains 2 missing records (`?`).
- Missing values are imputed strictly within training folds using median/mode estimators to avoid data leakage.
