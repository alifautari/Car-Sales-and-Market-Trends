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
- Univariate Analysis of Numerical Feature
  ```python
  num_columns=df.select_dtypes(include=np.number).columns.to_list()
  plt.figure(figsize=(14,10))
  plt.suptitle("Univariate Analysis of Numerical Features",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
  for i, col in enumerate (num_columns):
    plt.subplot(3,2,i+1)
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f"Distribution of {col}")
  plt.tight_layout(rect=[0, 0, 1, 0.99])
  plt.savefig("01_Univariate_Analysis_of_ Numerical_Features.jpg", bbox_inches='tight')
  plt.show()
  ```
![Univariate Analysis of Numerical Features](output/figure/01_Univariate_Analysis_of_ Numerical_Features.jpg)
- Univariate Analysis of Categorical Feature
  ```python
  cat_columns=['make','body', 'transmission','state', 'color', 'interior']
  plt.figure(figsize=(20,18))
  plt.suptitle("Univariate Analysis of Categorical Features",fontsize=20,fontweight='bold',alpha=0.8,y=1.)
  for i, col in enumerate (cat_columns):
    plt.subplot(3,2,i+1)
    sns.countplot(x=df[col], order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"Distribution of {col}")
  plt.tight_layout(rect=[0, 0, 1, 0.99])
  plt.savefig("02_Univariate_Analysis_of_Categorical_Features.jpg", bbox_inches='tight')
  plt.show()
  ```
  ![Univariate Analysis of Categorical Features](output/figure/02_Univariate_Analysis_of_Categorical_Features.jpg)
- Multivariate Analysis (Correlation of Numerical Features)
  ```python
  plt.figure(figsize=(8,5))
  sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="crest")
  plt.title("Multivariate Analysis (Correlation of Numerical Features)")
  plt.savefig("03_Multivariate_Analysis_(Correlation_of_Numerical_Features).jpg", bbox_inches='tight')
  plt.show()
  ```
  ![Multivariate Analysis (Correlation of Numerical Features)](output/figure/03_Multivariate_Analysis_(Correlation_of_Numerical_Features).jpg)
- Distribution of car prices
  ```python
  if 'make' in df.columns:
    plt.figure(figsize=(14,7))
    sns.boxplot(x='make', y='sellingprice', data=df, palette='flare')
    plt.xticks(rotation=90)
    plt.title("Brand vs Selling Price")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("04_Brand_vs_Selling_Price_(boxplot).jpg", bbox_inches='tight')
    plt.show()
  ```
  ![Brand vs Selling Price)](output/figure/04_Brand_vs_Selling_Price_(boxplot).jpg)
- Relationship between odometer and price
  ```python
  if 'odometer' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='odometer', y='sellingprice', data=df, hue='body', palette='magma')
    plt.title("Odometer vs Selling Price")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("05_Odometer_vs_Selling_Price_(scatterplot).jpg", bbox_inches='tight')
    plt.show()
  ```
  ![(Odometer vs Selling Price)](output/figure/05_Odometer_vs_Selling_Price_(scatterplot).jpg)
- Effect of condition and MMR
  ```python
  if 'condition' in df.columns:
    plt.figure(figsize=(8,5))
    sns.lineplot(x='condition', y='mmr', data=df)
    plt.title("Cars' Conditions vs Manheim Market Report (MMR)")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("06_Condition_vs_MMR.jpg", bbox_inches='tight')
    plt.show()
  ```
  ![(Condition vs MMR)](output/figure/06_Condition_vs_MMR.jpg)

## Visualization
- Trends of Monthly Sales Revenue (Top 10 Brands)
  ```python
  #Parsing datetime
  df['date'] = pd.to_datetime(df['saledate'], errors="coerce").dt.tz_convert(None)
  
  #Group by saledate (monthly) and brand, then sum selling price
  group = df.groupby([
    df['date'].dt.to_period('M').dt.to_timestamp(),  # group by month
    'make'
  ])['sellingprice'].sum().reset_index()

  # Top 10 brands by total selling price
  top_brand = (
    group.groupby('make')['sellingprice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
  )
  
  # Filter grouped data to only include those brands
  top_group = group[group['make'].isin(top_brand)]
  
  # Convert sales to millions
  top_group.loc[:, 'Sales (Million)'] = top_group['sellingprice'] / 1e6
  
  # Plot
  plt.figure(figsize=(12, 6))
  sns.lineplot(data=top_group, x='date', y='Sales (Million)', hue='make', marker=(8,0,0))
  plt.title('Monthly Sales Revenue (Top 10 Brands)', fontsize=20)
  plt.xlabel('Sale Date')
  plt.ylabel('Sales (Million, $)')
  ax = plt.gca()
  ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))  # every 1 month
  ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # format as Jan 2015
  plt.xticks(rotation=45)
  plt.legend(title='Brand', bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.tight_layout(rect=[0, 0, 1, 0.99])
  plt.grid(True)
  plt.savefig("07_Monthly_Sales_Revenue_(Top_10_Brands).jpg", bbox_inches='tight')
  plt.show()
  ```
  ![(Monthly Sales Revenue (Top 10 Brands))](output/figure/07_Monthly_Sales_Revenue_(Top_10_Brands).jpg)
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
