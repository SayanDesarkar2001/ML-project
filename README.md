# Energy Consumption Data Analysis and Visualization using ML

**Columms and their description:**

1. **customer_id**: Unique identifier for each customer.
2. **region**: Geographic region of the customer.
3. **energy_consumption_kwh**: Total energy consumption in kilowatt-hours.
4. **peak_hours_usage**: Energy usage during peak hours.
5. **off_peak_usage**: Energy usage during off-peak hours.
6. **renewable_energy_pct**: Percentage of energy from renewable sources.
7. **billing_amount**: Total billing amount.
8. **household_size**: Number of people in the household.
9. **temperature_avg**: Average temperature.
10. **income_bracket**: Income bracket of the household.
11. **smart_meter_installed**: Whether a smart meter is installed.
12. **time_of_day_pricing**: Whether time-of-day pricing is used.
13. **annual_energy_trend**: Annual trend in energy consumption.
14. **solar_panel**: Whether solar panels are installed.
15. **target_high_usage**: Whether the household is targeted for high usage.

**Operations performed on the dataset:**

1.  **Data Loading (pd.read_csv()):**
    * The code starts by loading the energy dataset from a CSV file into a pandas DataFrame. This is the standard way to ingest tabular data in Python for data analysis and machine learning.

2.  **Outlier Removal (remove_outliers() function):**
    * A custom function remove_outliers is defined to identify and remove outliers from specified numerical columns using the Interquartile Range (IQR) method.
    * For each specified column:
        * The first quartile (Q1) and third quartile (Q3) are calculated.
        * The IQR (Q3 - Q1) is computed.
        * Lower and upper bounds are determined as Q1 - 1.5 * IQR and Q3 + 1.5 * IQR, respectively.
        * Data points outside these bounds are filtered out, effectively removing the outliers.

3.  **Exploratory Data Analysis (EDA) and Visualization (matplotlib.pyplot, seaborn):**
    * The code includes several steps for visualizing the data to understand its characteristics:
        * **df.hist():** Generates histograms for all numerical features to visualize their distributions.
        * **sns.boxplot():** Creates box plots for numerical features to visually identify outliers and understand the spread of the data.
        * **sns.pairplot():** Displays pairwise relationships between numerical features, helping to identify potential correlations and patterns.
        * **sns.heatmap():** Generates a heatmap of the correlation matrix between numerical features, quantifying the linear relationships between them.
        * **df.groupby(...).plot(kind='bar'):** Creates bar plots to visualize aggregated data based on categorical features like 'region' and 'income\_bracket'. This helps in understanding the relationship between these categories and energy consumption or billing amount.
        * **df['household_size'].value_counts().plot(kind='bar'):** Shows the distribution of household sizes.
        * These visualizations provide insights into the data's distribution, potential outliers (though outliers are removed before this), and relationships between variables.
   <img width="741" alt="image" src="https://github.com/user-attachments/assets/e56cc9ff-d96e-4d39-9c87-2fc3efd3aeee" />
   <img width="769" alt="Screenshot 2025-04-22 131552" src="https://github.com/user-attachments/assets/dc65a432-a681-4686-ba4e-c9e001a1ee4b" />

4.  **Data Splitting (train_test_split()):**
    * The dataset is split into training and testing sets. The test_size=0.3 indicates that 30% of the data is reserved for testing, and random_state=42 ensures reproducibility of the split.
    * The test set is further split into a validation set (for initial model evaluation as a proxy for tuning) and an unseen test set for the final evaluation of the trained pipelines.

5.  **Feature Scaling (StandardScaler()):**
    * StandardScaler from sklearn.preprocessing is used to standardize the numerical features. This involves:
        * Calculating the mean and standard deviation of each feature in the training set.
        * Subtracting the mean and dividing by the standard deviation for each data point in the training, validation, and unseen test sets.
    * Scaling is important for many machine learning algorithms as it prevents features with larger values from dominating those with smaller values and can help with the convergence of some models.

6.  **Initial Model Training and Evaluation (Linear Regression, Random Forest, Decision Tree):**
    * Basic instances of LinearRegression, RandomForestRegressor, and DecisionTreeRegressor are initialized.
    * Each model is trained on the scaled training data (X_train_scaled, y_train).
    * Predictions are made on the scaled validation data (X_val_scaled).
    * The performance of each model is evaluated using Mean Squared Error (mean_squared_error) and R-squared (r2_score).
   <img width="707" alt="image" src="https://github.com/user-attachments/assets/252dfa12-3789-475c-9056-6f760121fa91" />


7.  **Hyperparameter Tuning with GridSearchCV() (for Random Forest and Decision Tree):**
    * GridSearchCV is used to systematically search for the best combination of hyperparameters for the Random Forest and Decision Tree models.
    * **Pipeline Creation (Pipeline()):** For both models, a Pipeline is created that first applies StandardScaler and then the respective regressor. This ensures that scaling is applied to the data during the cross-validation process within GridSearchCV.
    * **Parameter Grids (rf_param_grid, dt_param_grid):** Dictionaries defining the hyperparameters and their possible values to be tested by GridSearchCV.
    * **Fitting GridSearchCV:** The fit() method of GridSearchCV trains and evaluates the model for every combination of hyperparameters in the grid using cross-validation (here, cv=3).
    * **Best Model Extraction (best_estimator_):** The best_estimator_ attribute of the fitted GridSearchCV object provides the model (pipeline in this case) with the best hyperparameters found.
   <img width="693" alt="image" src="https://github.com/user-attachments/assets/35ff2c57-b54f-4c8b-9ce6-4a96e589f014" />

8.  **Evaluation of Tuned Models on Unseen Data:**
    * The best_pipeline (the tuned model within the pipeline, including the scaler) for both Random Forest and Decision Tree is used to make predictions on the scaled unseen test data (X_unseen_scaled).
    * The performance on this truly held-out data is evaluated using MSE and R-squared to estimate the model's generalization ability.

9.  **Model Saving (joblib.dump()):**
    * The joblib library is used to save the trained best_pipeline for both Random Forest and Decision Tree to disk. This allows you to load and reuse the trained models later without retraining.

10. **Loading and Evaluating Saved Models (for verification):**
    * The code demonstrates how to load the saved pipelines using joblib.load().
    * The loaded pipelines are then used to make predictions on the unseen data, and the performance is evaluated again to ensure consistency.
