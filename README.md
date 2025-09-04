# Car-Sales-and-Market-Trends
This project combines **exploratory data analysis (EDA)** and **machine learning modeling** to understand the used car market and predict car selling prices.

## Objectives
1. Perform data cleaning and preprocessing  
2. Explore key insights (trends, correlations, distributions)  
3. Build predictive models (Linear Regression)  
4. Evaluate model performance  

## Dataset
- Source: [Kaggle - Vehicle Sales Data](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data)  
- Size: 500,000+ records  
- Features: year, brand, model, trim, body, transmission, state, condition, odometer, color, interior, seller, mmr, selling price, saledate.  

## Tools
- Python (pandas, numpy, matplotlib, seaborn, plotly, scikit-learn)  
- Jupyter Notebook  
- Tableau (for dashboard visualization)  

## Exploratory Data Analysis (EDA)
- Univariate Analysis (Histogram) of Numerical Feature '''python num_columns=df.select_dtypes(include=np.number).columns.to_list()
plt.figure(figsize=(14,10))
plt.suptitle("Univariate Analysis of Numerical Features",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
for i, col in enumerate (num_columns):
    plt.subplot(3,2,i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig("01_Univariate_Analysis_of_ Numerical_Features.jpg", bbox_inches='tight')
plt.show()'''   ![Univariate Analysis of Numerical Features](output/Univariate%20Analysis%20of%20Numerical%20Features.jpg)
- Univariate Analysis (Bar) of Categorical Feature ![Univariate Analysis of Categorical Features](output/Univariate%20Analysis%20of%20Categorical%20Features.jpg)
- Multivariate Analysis (Correlation of Numerical Features) ![Multivariate Analysis (Correlation of Numerical Features)](output/Multivariate%20Analysis%20(Correlation%20of%20Numerical%20Features).jpg)
- Distribution of car prices (boxplot)  
- Relationship between odometer and price  
- Effect of condition and MMR

## Visualization
- Trends of Monthly Sales Revenue (Top 10 Brands)
- Top 10 Most Popular Cars (Brand-Model)
- Top 10 Cars by Total Sales Revenue
- Treemap of top cars models by total revenue
- Scatterplot of Car Selling Price vs Manufacturing Year by Brand

## Modeling
- Linear Regression  
- Metrics: MAE, RMSE, R²  

## Results
- Key insights: Car prices are mainly influenced by condition and MMR.  

## Dashboard
👉 Tableau Dashboard: [View Here](https://public.tableau.com/...)  
