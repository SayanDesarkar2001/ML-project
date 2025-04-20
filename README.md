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

**Manipulations performed on dataset:**

1. **Loading the Dataset**: The dataset is loaded from a CSV file.
2. **Displaying Initial Shape**: The shape of the dataset is displayed before any manipulations.
3. **Removing Outliers**: Outliers are removed from specified columns using the Interquartile Range (IQR) method.
4. **Displaying Final Shape**: The shape of the dataset is displayed after outlier removal.
5. **Displaying Data Types**: The data types of the features are displayed using the `.info()` method.
6. **Visualizing Data**:
   - **Histograms**: Histograms of numerical features are plotted to show their distributions.
   - **Boxplot**: A boxplot is created to visualize the presence of outliers in numerical features.
   - **Pairplot**: Pairwise relationships between numerical features are visualized using a pairplot.
   - **Heatmap**: A heatmap is generated to show the correlation between different numerical features.
7. **Additional Visualizations**:
   - **Total Energy Consumption by Region**: A bar plot showing total energy consumption by region.
   - **Average Renewable Energy Percentage by Region**: A bar plot showing the average renewable energy percentage by region.
   - **Average Billing Amount by Income Bracket**: A bar plot showing the average billing amount by income bracket.
   - **Household Size Distribution**: A bar plot showing the distribution of household sizes.
   - **Smart Meter Installation by Region**: A bar plot showing the number of smart meters installed by region.
   - **Time of Day Pricing Adoption by Region**: A bar plot showing the number of households with time-of-day pricing by region.
