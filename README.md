# SME BANKRUPTCY PREDICTION USING MACHINE LEARNING

## ABOUT

The SME Bankruptcy Prediction System is a specialized machine learning solution designed to assess the financial stability of Small and Medium Enterprises (SMEs). Recognizing that early detection of financial distress is crucial for economic stability, this system leverages the Extreme Gradient Boosting (XGBoost) algorithm to analyze historical financial ratios and predict the likelihood of insolvency. By automating the risk assessment process, the system provides lenders, investors, and stakeholders with a data-driven tool to mitigate financial losses and facilitate informed credit decisions.

## FEATURES

* **Batch Data Processing:** Capable of ingesting and analyzing large datasets via CSV/Excel uploads for simultaneous risk assessment of multiple companies.
* **Manual Risk Calculator:** A dedicated interface for ad-hoc analysis, allowing users to input specific financial metrics for single-entity evaluation.
* **Advanced Predictive Modeling:** Utilizes the XGBoost classifier, optimized for structured financial data, ensuring high classification accuracy.
* **Imbalance Handling:** Implements Synthetic Minority Over-sampling Technique (SMOTE) to address class imbalance, ensuring robust detection of minority bankruptcy cases.
* **Automated Reporting:** Generates comprehensive PDF risk reports detailing the prediction status, probability scores, and key contributing factors.
* **Real-time Dashboard:** Provides immediate visualization of risk probabilities and financial health indicators.

## REQUIREMENTS

* **Operating System:** Windows 10/11 (64-bit) or Ubuntu Linux
* **Development Environment:** Python 3.8 or later
* **Machine Learning Libraries:** Scikit-learn, XGBoost, Imbalanced-learn (SMOTE)
* **Data Processing:** Pandas, NumPy
* **Web Framework:** Flask or Streamlit
* **Visualization:** Matplotlib, Seaborn
* **Report Generation:** FPDF or ReportLab
* **Version Control:** Git

## SYSTEM ARCHITECTURE

The system is composed of the following functional modules:
1. **Data Ingestion Module:** Handles the upload and parsing of financial datasets (CSV/Excel) and validates input format and data integrity.
2. **Preprocessing & Balancing Module:** Cleans data by handling missing values and outliers, and applies SMOTE to balance the dataset between solvent and bankrupt classes.
3. **Predictive Engine:** An optimized XGBoost classifier trained on historical financial ratios, generating probability scores and binary classification outcomes.
4. **Application Interface:** A web-based dashboard for user interaction and result visualization.
5. **Reporting Module:** Compiles analysis results into formal PDF documents for archival and presentation.

## PROJECT DIRECTORY STRUCTURE

<img width="541" height="436" alt="image" src="https://github.com/user-attachments/assets/868e6de6-27a2-4c19-b40a-78d1317af6b2" />


## DATASET DESCRIPTION

The model is trained on a comprehensive dataset containing financial ratios from Small and Medium Enterprises. The key features used for prediction include:

* **Net Income to Total Assets:** Measures how efficiently the company uses its assets to generate profit.
* **Total Liabilities to Total Assets:** Indicates the percentage of the company's assets that are financed by debt (Levierage).
* **Working Capital to Total Assets:** Reflects the company's short-term financial health and liquidity.
* **Current Assets to Short-term Liabilities:** A rigorous test of liquidity (Current Ratio).
* **EBIT to Total Assets:** Earnings Before Interest and Taxes relative to asset base.
* **Retained Earnings to Total Assets:** Indicates the extent to which the company relies on self-financing.

The dataset underwent rigorous preprocessing, including normalization of numerical values and SMOTE augmentation to correct the imbalance between solvent and bankrupt class samples.

## OUTPUT AND RESULTS

The system output includes a real-time status display indicating "Solvent" or "High Risk," accompanied by a precise probability percentage. Additionally, users can download a structured PDF report that summarizes the company profile, input ratios, and the final risk assessment.

<img width="1068" height="486" alt="image" src="https://github.com/user-attachments/assets/a3b013a1-2e54-4272-bd20-d67421459c9b" />

<img width="1102" height="688" alt="Screenshot 2025-12-22 233848" src="https://github.com/user-attachments/assets/3de23bf0-de5c-4e6a-8314-c520b8bba0ab" />

<img width="1025" height="868" alt="Screenshot 2025-12-22 233440" src="https://github.com/user-attachments/assets/5a17b201-1ce8-4f24-8a57-4c31e3af951d" />

<img width="1910" height="863" alt="Screenshot 2025-12-19 224838" src="https://github.com/user-attachments/assets/3898ed96-5d3c-4fd7-b619-3cf8d7fd6f79" />



* **Detection Accuracy:** The model demonstrates high precision (~94%) in identifying potential bankruptcy cases, significantly outperforming traditional statistical methods.
* **Financial Utility:** Reduces the risk of non-performing loans (NPLs) for financial institutions by providing early warning signals.
* **Scalability:** The architecture supports the integration of additional financial indicators and can be scaled to handle larger datasets as required.

## FUTURE ENHANCEMENTS

While the current system offers robust predictive capabilities, the following enhancements are envisioned for future iterations:

* **Integration of Macroeconomic Variables:** Incorporating external factors such as inflation rates, GDP growth, and market volatility to refine risk assessments.
* **Explainable AI (XAI) Integration:** Implementing SHAP (SHapley Additive exPlanations) values to provide granular explanations for why a specific company was classified as high risk.
* **Cloud Deployment:** Migrating the application to a cloud infrastructure (AWS or Azure) to support high-availability and scalable data processing.

## REFERENCES

1. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016.
2. E. I. Altman, "Financial ratios, discriminant analysis and the prediction of corporate bankruptcy," The Journal of Finance, vol. 23, no. 4, pp. 589–609, 1968.
3. N. V. Chawla, K. W. Bowyer, L. O. Hall, and W. P. Kegelmeyer, "SMOTE: Synthetic Minority Over-sampling Technique," Journal of Artificial Intelligence Research, vol. 16, pp. 321–357, 2002.
4. M. Ziogas and N. Metaxiotis, "Bankruptcy prediction of Small and Medium Enterprises (SMEs) using Machine Learning algorithms," International Journal of Decision Support Systems, 2020.
5. P. Kou, Y. Yang, and Z. Peng, "SME bankruptcy prediction with XGBoost and SMOTE: An empirical study," IEEE International Conference on Industrial Engineering and Engineering Management (IEEM), 2021.

## CONTRIBUTORS

This project was developed as a requirement for the academic curriculum.
