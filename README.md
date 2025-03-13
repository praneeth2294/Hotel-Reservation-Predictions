# Hotel-Reservation-Predictions

## Project Overview
This project aims to analyze hotel booking data and build predictive models to determine whether a hotel reservation will be canceled. By leveraging machine learning (ML) and artificial neural networks (ANN), we identify key factors influencing cancellations, helping hotels optimize their operations and reduce revenue loss.

## Dataset
* Total records: 36,275
* Total features: 19 (including booking details, customer demographics, and preferences)
* Target variable: booking_status (Canceled or Not_Canceled)
  
## Objectives
* Perform Exploratory Data Analysis (EDA) to uncover patterns in the data.
* Apply data preprocessing (handling missing values, removing outliers, encoding categorical features, and scaling numerical data).
* Train various machine learning models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).
* Implement Artificial Neural Networks (ANN) for better predictive accuracy.
* Evaluate models using accuracy, precision, recall, and F1-score.

## Exploratory Data Analysis (EDA)
### Data Cleaning
* Handled missing values and categorical encoding.
* Removed outliers using Winsorization for no_of_previous_cancellations, no_of_week_nights, and no_of_weekend_nights.

### Feature Engineering
* Label encoding for type_of_meal_plan, room_type_reserved, and market_segment_type.
* One-hot encoding for booking_status.
* Applied Robust Scaling to handle non-normal distributions and outliers.

## Model Performance

| Model                 | Training Accuracy | Testing Accuracy |
|-----------------------|------------------|-----------------|
| Logistic Regression   | 95%              | 85%             |
| Decision Tree        | 99%              | 84%             |
| Random Forest       | 98%              | 86%             |
| Gradient Boosting   | 96%              | 86%             |
| XGBoost             | 97%              | 87%             |
| ANN (Neural Network) | 87%              | 87%             |

## Deep Learning Model (ANN)
The ANN model architecture:

* Input Layer: 256 neurons (ReLU)

* Hidden Layers: 3 layers with 256 neurons each (ReLU + Dropout)

* Output Layer: 1 neuron (Sigmoid)

* Optimizer: Adam

* Loss Function: Binary Cross-Entropy

* Training Accuracy: 87%

* Testing Accuracy: 87%

## Key Insights
* Most bookings were made in 2018.
* Many customers booked rooms through online platforms.
* Lead time, room type, previous cancellations, and market segment type were strong predictors of cancellation.
* The ANN model performed the best, achieving 87% accuracy in both training and testing phases.

## Future Improvements
* Fine-tune hyperparameters using Grid Search or Bayesian Optimization.
* Incorporate time-series forecasting for seasonal trends.
* Explore additional deep learning architectures like LSTMs for sequential data
