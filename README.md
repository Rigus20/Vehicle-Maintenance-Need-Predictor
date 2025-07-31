# Vehicle-Maintenance-Need-Predictor

> **Project Objective**
>The goal is to predict **`Need_Maintenance`**—a binary flag that indicates whether a vehicle currently requires maintenance (`1 = Yes`, `0 = No`). The original dataset contains **5,000 rows × 20 columns**

---
## Dataset Overview

The dataset records **vehicle condition, usage, and ownership details** for a fleet of mixed vehicle types.  


### Feature Dictionary

| Feature | Type | Description |
|---------|------|-------------|
| `Vehicle_Model` | Categorical | Type of vehicle (Car, SUV, Van, Truck, Bus, Motorcycle). |
| `Mileage` | Numeric | Total distance travelled (km). |
| `Maintenance_History` | Ordinal | Historical service quality (Good > Average > Poor). |
| `Reported_Issues` | Numeric | Count of issues reported by drivers or diagnostics. |
| `Vehicle_Age` | Numeric | Vehicle age (years). |
| `Fuel_Type` | Categorical | Primary fuel (Diesel, Petrol, Electric). |
| `Transmission_Type` | Categorical | Gearbox type (Automatic, Manual). |
| `Engine_Size` | Numeric | Engine displacement (cc). |
| `Odometer_Reading` | Numeric | Current odometer reading (km). |
| `Last_Service_Date` | Date | Date of the last completed service. |
| `Warranty_Expiry_Date` | Date | Warranty end date. |
| `Owner_Type` | Categorical | Ownership history (First, Second, Third owner). |
| `Insurance_Premium` | Numeric | Annual insurance premium (monetary unit). |
| `Service_History` | Numeric | Number of scheduled services performed. |
| `Accident_History` | Numeric | Count of recorded accidents. |
| `Fuel_Efficiency` | Numeric | Fuel economy (km per liter). |
| `Tire_Condition` | Ordinal | Tire state (New > Good > Worn Out). |
| `Brake_Condition` | Ordinal | Brake state (New > Good > Worn Out). |
| `Battery_Status` | Ordinal | Battery health (New > Good > Weak). |
| `Need_Maintenance` | **Target** | Binary label—`1` if maintenance is required, `0` otherwise. |
---

## 1  Data Preparation

* **Categorical encoding**  — Levels are explicitly set and then converted to numeric codes to facilitate modelling.
* **Imbalance correction**  — Because the target was skewed, we create a **balanced subset** with **500 positives** and **500 negatives** to avoid over‑fitting.

---

## 2  Model Building

1. **Manual covariate pruning** guided by BIC - this sub-step proved ineffective in practice.
2. **Bidirectional stepwise selection** based on **AIC**.

The optimal model (AIC = 632.92) retains:

```
Need_Maintenance ~ Mileage + Maintenance_History + Reported_Issues +
                   Vehicle_Age + Fuel_Type + Owner_Type + Service_History +
                   Accident_History + Fuel_Efficiency + Brake_Condition +
                   Battery_Status
```

---

## 3  Exploratory Analysis

### 3.1  Pairwise Relationships

<img width="1408" height="800" alt="ggpairs" src="https://github.com/user-attachments/assets/4e1e7f0b-2401-453c-a3f3-4d9b7e1b33fa" />

The plot highlights both positive (corr > 0) and negative (corr < 0) associations with **Need\_Maintenance**.

### 3.2  Key Covariate Trends

* **Reported\_Issues vs Need\_Maintenance**  — A clear upward trend: more reported issues → higher probability of maintenance.
<img width="1003" height="620" alt="image" src="https://github.com/user-attachments/assets/94b9098f-015e-4bfb-97c2-95cd663d253d" />

* **Vehicle\_Age vs Need\_Maintenance**

  * Early years: high maintenance probability, likely due to recalls/defects.
  * Middle years: dip followed by an increase caused by wear‑and‑tear.
  * Older than ≈ 10 years: slight decrease; very old cars in the dataset appear mechanically resilient.

![1000171364](https://github.com/user-attachments/assets/0544b79f-63a0-4959-801c-946eb3f82ec5)


* **Accident\_History vs Need\_Maintenance**  — Fairly flat until a slight rise at 3 accidents; beyond that the vehicle is usually written off, so no further data are collected.
<img width="1003" height="620" alt="image" src="https://github.com/user-attachments/assets/916e699f-1879-4a19-9c48-63e89fdc0a25" />


---

## 4  Collinearity Check

| Predictor            | VIF      |
| -------------------- | -------- |
| Mileage              | 1.03     |
| Maintenance\_History | 1.26     |
| Reported\_Issues     | **1.64** |
| Vehicle\_Age         | 1.04     |
| Fuel\_Type           | 1.02     |
| Owner\_Type          | 1.03     |
| Service\_History     | 1.04     |
| Accident\_History    | 1.03     |
| Fuel\_Efficiency     | 1.05     |
| Brake\_Condition     | 1.40     |
| Battery\_Status      | 1.40     |

All VIFs < 5 → negligible multicollinearity; *Reported\_Issues* shows the highest but still acceptable value.

---

## 5  Odds Ratios (per 10‑unit change)

| Predictor            | OR          |
| -------------------- | ----------- |
| Mileage              | 0.9999      |
| Maintenance\_History | 1.4 × 10⁻⁵  |
| Reported\_Issues     | 5.1 × 10⁵   |
| Vehicle\_Age         | 1.66        |
| Fuel\_Type           | 6.95        |
| Owner\_Type          | 14.11       |
| Service\_History     | 4.75        |
| Accident\_History    | 8.82        |
| Fuel\_Efficiency     | 0.49        |
| Brake\_Condition     | 9.9 × 10⁻¹⁰ |
| Battery\_Status      | 1.8 × 10⁻⁹  |

**How to read these odds ratios**

An odds ratio (OR) expresses how the *odds* of the vehicle needing maintenance change for a **10‑unit increase** in the predictor (or a one‑level shift for ordered factors), keeping every other variable constant:

* **OR > 1** — odds **increase**. For instance, moving from a first‑ to a third‑hand car (**Owner\_Type**) multiplies the odds by about **14 ×**.
* **OR < 1** — odds **decrease**. A 10‑unit gain in **Fuel\_Efficiency** nearly halves the odds (OR = 0.49).
* **OR ≈ 1** — virtually no effect.

**Key take‑aways**

* **Reported\_Issues** dominates the model (OR ≈ 5.1 × 10⁵): each additional logged complaint skyrockets the likelihood of imminent maintenance.
* Solid **Maintenance\_History** is protective (OR ≈ 1.4 × 10⁻⁵) — well‑serviced vehicles are far less likely to need extra work.
* Component quality matters: newer **Brake\_Condition** and **Battery\_Status** drive ORs far below 1, confirming that fresh parts reduce risk.

---

## 6  Model Evaluation

* **Binned residuals plot** — Good overall fit; minor tails remain.
<img width="1003" height="620" alt="image" src="https://github.com/user-attachments/assets/b5de8d2f-1224-4cc0-8f6d-44fd5519b94d" />

* **Hosmer–Lemeshow test** — p‑value = 0.3482 → model is not rejected.
* **ROC curve** — AUC = 0.9409 → excellent discrimination.
<img width="1003" height="620" alt="image" src="https://github.com/user-attachments/assets/4e78ec51-daf9-4930-9599-6ddcd0cc31a5" />


---

## 7  Leave‑One‑Out Cross‑Validation

Dataset split: **80 % train / 20 % test**.

* Polynomial degrees 1–4 tested.
* **2nd‑degree** offers the lowest LOOCV error with minimal computational cost (tied with degree 3).
<img width="1003" height="620" alt="image" src="https://github.com/user-attachments/assets/5a1345ad-f1a7-419e-ac4e-cffc9d71e943" />

---

## 8  Test‑Set Performance (threshold = 0.40)

|            | **Actual 0** | **Actual 1** |
| ---------- | ------------ | ------------ |
| **Pred 0** | 85           | 11           |
| **Pred 1** | 15           | 89           |

* **Precision:** 0.856
* **Recall (Sensitivity):** 0.890
* **F1‑Score:** 0.873
* **Accuracy:** 0.870

### Metric Definitions

* **Precision** — Proportion of predicted positives that are correct.
* **Recall** — Share of true positives that are found.
* **F1‑Score** — Harmonic mean of Precision and Recall.
* **Accuracy** — Overall proportion of correct predictions.

---

## 9  Conclusions

The logistic model—with balanced data, careful feature selection, and thorough diagnostics—achieves **AUC ≈ 0.94** and **Accuracy ≈ 0.87**. These results indicate a robust predictor capable of balancing false positives and true positives effectively.
