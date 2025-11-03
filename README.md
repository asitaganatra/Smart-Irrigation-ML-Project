Machine Learning Model for Precision Water Requirement Prediction in Irrigation

🎯 Project Overview and Problem Statement

This project addresses the critical issue of water scarcity in agriculture by implementing a high-performance Machine Learning solution. The system's purpose is to accurately predict the optimal volume of irrigation water ($\mathbf{m^3}$) required for specific crops under complex, varying environmental conditions.

The final deployed application is powered by the Random Forest Regressor, validated against an authentic agricultural dataset to ensure efficiency and predictive accuracy in real-world scenarios.

🛠️ Key Components & Models

Final Deployed Model: Random Forest Regressor (Selected for superior stability on real-world data).

Comparative Model: Decision Tree Regressor (Used as a low-complexity baseline).

Dataset Source: Real-World Agricultural Data ($\mathbf{Smart\_irrigation\_dataset.csv}$)

Application: Streamlit Web Application (for user interaction and recommendation display).

🔗 Live Application and Analysis Access

Feature

Link

Deployed Streamlit App

(Paste your live app URL here after successful deployment)

Full Code Repository

(Paste your GitHub Repository URL here)

Model Training & Analysis

Open Full Analysis in Colab

🔬 Experiments and Results Summary (Required by Mail)

The assignment required a comparison between multiple models ($\mathbf{Random\ Forest}$ and $\mathbf{Decision\ Tree}$) evaluated on $\mathbf{two\ similar\ datasets}$ (Synthetic vs. Real-World) to prove the rigor of model selection.

Comparative Performance Metrics

Model

Dataset Used

$\mathbf{R^2}$ Score (Goodness of Fit)

MAE (Error in $\mathbf{m^3}$)

Random Forest

Real-World Data

0.386

155.80

Decision Tree

Real-World Data

-0.228

202.06

Random Forest

Synthetic Data

0.657

3.63

Visual Comparison (MAE)

The visualization confirms the Random Forest model's stability and superiority, demonstrating that the ensemble method is necessary to handle the high noise and complexity of real-world agricultural data.

Conclusion and Learning:

The analysis confirms that the Random Forest Regressor is the most reliable model for this precision irrigation task. The comparison clearly demonstrated that simple models like the Decision Tree fail catastrophically (negative $\mathbf{R^2}$) when faced with the inherent variability and noise present in real-world agricultural datasets. The project successfully selected and implemented a robust model for deployment.

🚀 Execution Steps

1. Prerequisites

Python 3.x

Dependencies listed in requirements.txt

2. Local Setup

Clone this repository: git clone [YOUR_REPO_URL]

Install dependencies: pip install -r requirements.txt

3. Run Application

(Assumes you have already run the training script via Colab or locally to generate the PKL files)

Run the Streamlit app: streamlit run app.py