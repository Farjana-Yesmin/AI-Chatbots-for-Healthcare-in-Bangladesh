AI Chatbot for Dengue Symptom Triage in Bangladesh 🌡️🤖

https://img.shields.io/badge/DASGRI-2026-brightgreen
https://img.shields.io/badge/Springer-LNNS-blue

Exciting News! 🎉 This paper has been accepted for presentation at DASGRI-2026 and publication in Springer's Lecture Notes in Networks and Systems series (Top 15% of submissions).


Overview

This project develops an AI-powered chatbot for dengue symptom assessment and triage in Bangladesh, using machine learning models trained on demographic and clinical data. The system provides personalized health advice in both English and Bangla, helping to alleviate healthcare strain during dengue outbreaks.

Key Features

🤖 Multilingual Chatbot: Supports English and Bangla with symptom assessment
🌡️ Severity Prediction: Decision Tree classifier trained on 1,000+ dengue cases (2019-2023)
📊 Comprehensive Analysis: Feature importance, correlation studies, and bias mitigation
🌦️ Climate Integration: Incorporates temperature and rainfall data for enhanced predictions
⚖️ Ethical Design: Public data only, no PII, bias mitigation via SMOTE and fairness checks

Model Performance

Model	Accuracy	F1-Score	CV F1
Basic Decision Tree	1.000*	1.000*	1.000*
Non-Leaky Subset DT	0.790	0.802	0.529
External Simulation	-	0.995	-
*Full dataset with potential leakage from serology markers

Dataset Information

Primary Dataset: 1,000 dengue cases (2019-2023) from Bangladesh
Features: Demographics (Age, Gender, AreaType, HouseType, District) + Clinical markers (NS1, IgG, IgM)
Climate Data: Temperature and rainfall patterns integrated for ecological analysis
Ethical Compliance: Public data, no PII, IRB exempt (Boise State #2025-001)

Installation & Usage

Prerequisites


pip install pandas numpy scikit-learn imbalanced-learn nltk matplotlib seaborn
Quick Start


# Load and preprocess data
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Train the model
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Use the chatbot
response = enhanced_chatbot("Rahim", 25, "Male", 1, 0, 1, "Developed", "Building", "Dhaka", lang='en')
print(response)
Project Structure

text
├── data/                    # Dataset files (see Data Availability)
├── models/                  # Trained model files
├── notebooks/               # Jupyter notebooks for analysis
├── src/
│   ├── preprocessing.py    # Data cleaning and feature engineering
│   ├── training.py         # Model training and evaluation
│   └── chatbot.py          # Multilingual chatbot implementation
├── results/                # Output figures and performance metrics
└── README.md


Key Results

Feature Importance: Age is the dominant predictor (68.6% importance)
Fairness: Minimal bias across gender (Δ < 0.05) and age groups
Pilot Testing: 75% satisfaction rate (n=50 simulated users)
Clinical Utility: 28% referral rate for low-confidence cases
Ethical Considerations

✅ No PII collected
✅ Public dataset usage
✅ Bias mitigation via SMOTE
✅ Fairness validation across demographics
✅ Clear disclaimers for medical advice



Citation

If you use this work, please cite:

bibtex
@inproceedings{yesmin2026ai,
  title={AI Chatbots for Dengue Symptom Triage in Bangladesh: A Decision Tree Classifier Approach},
  author={Yesmin, Farjana},
  booktitle={Proceedings of DASGRI-2026},
  series={Lecture Notes in Networks and Systems},
  publisher={Springer},
  year={2026}
}

Data Availability

Dengue Severity Data: Government of Bangladesh HEOC Dashboard
Climate Correlation Data: Available on Kaggle
Processed Datasets: Contact author for access
Future Work

Integration with DGHS mobile applications
Real-world clinical trials (n=500 planned)
Expansion to other vector-borne diseases
Real-time climate data integration

Acknowledgments

This research was supported by public data from the Government of Bangladesh (HEOC Dengue Dashboard) and various Kaggle dataset contributors.

Contact: Farjana Yesmin | Conference: DASGRI-2026 | Publication: Springer LNNS

Part of the mission to make healthcare AI accessible and ethical in developing countries 🌍💚
