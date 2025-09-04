#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Data and Required Packages
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# In[3]:


import plotly.express as px
import plotly.io as pio  

# List of renderers 
preferred_renderers = ["notebook", "colab", "browser", "iframe"]

for r in preferred_renderers:
    try:
        pio.renderers.default = r
        break
    except Exception:
        continue


# In[4]:


import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[5]:


# Load The Dataset
df = pd.read_csv("car_prices.csv")
df.head()


# In[6]:


# Delete VIN (Vehicle Identification Number)
df.drop(columns=['vin'], axis=1, inplace=True)
df.head()


# In[7]:


# Check duplicate values
df.duplicated().sum()


# In[8]:


# Check Null Value Counts and DataTypes of the features
df.info()


# In[9]:


# Summary of Data
df.describe(include='all')


# In[10]:


# Data Cleaning & Preprocessing
# Check empty values
df.isnull().sum()


# In[11]:


# Fill missing values
df.fillna(df.median(numeric_only=True), inplace=True) #for numeric values: median

for col in df.select_dtypes(include=['object']).columns: #for categorical values: mode
    mode_val = df[col].mode()[0]   
    df[col].fillna(mode_val, inplace=True)


# In[12]:


# Convert to datetime
df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
df.head()


# In[13]:


# Check the number of unique values of each column
df.nunique()


# In[14]:


# Show Categories in columns
for i in df[['make','body', 'transmission','state', 'color', 'interior']]:
    print(f"The catagories in '{i}' are : ",list(df[i].unique()))


# In[15]:


# Text Standardization (string: lowercase and replace)
#'make' category
df['make'] = df['make'].str.strip().str.lower() 
make_replace = {
    "landrover": "land rover",
    "vw": "volkswagen",
    "mercedes" : "mercedes-benz",
    "mercedes-b": "mercedes-benz",
    "gmc truck": "gmc",
    "dodge tk": "dodge",
    "mazda tk": "mazda",
    "ford tk": "ford",
    "ford truck": "ford",
    "chev truck": "chevrolet"
}
df['make'] = df['make'].replace(make_replace)
print(f"The catagories in 'make' are : ",list(df['make'].unique()))


# In[16]:


#'body' category
df['body'] = df['body'].str.strip().str.lower() 
df['body'] = df['body'].replace({'navitgation': np.nan})
mode_body = df['body'].mode()[0]  
df['body'] = df['body'].fillna(mode_body)
body_replace = {
    # sedan
    "g sedan": "sedan",

    # convertible
    "g convertible": "convertible",
    "g37 convertible": "convertible",
    "beetle convertible": "convertible",
    "cts-v convertible": "convertible",
    "q60 convertible": "convertible",
    "granturismo convertible": "convertible",

    # coupe
    "g coupe": "coupe",
    "elantra coupe": "coupe",
    "genesis coupe": "coupe",
    "cts coupe": "coupe",
    "q60 coupe": "coupe",
    "g37 coupe": "coupe",
    "cts-v coupe": "coupe",
    "koup": "coupe",

    # wagon
    "cts wagon": "wagon",
    "tsx sport wagon": "wagon",
    "cts-v wagon": "wagon",

    # van / minivan
    "e-series van": "van",
    "transit van": "van",
    "promaster cargo van": "van",
    "ram van": "van",
    "minivan": "van",

    # cab (pickup body style)
    "crew cab": "cab",
    "double cab": "cab",
    "extended cab": "cab",
    "regular cab": "cab",
    "quad cab": "cab",
    "king cab": "cab",
    "crewmax cab": "cab",
    "access cab": "cab",
    "club cab": "cab",
    "xtracab": "cab",
    "cab plus": "cab",
    "cab plus 4": "cab",
    "mega cab": "cab",
    "supercrew": "cab",
    "supercab": "cab",
    "regular-cab": "cab"
}
df['body'] = df['body'].replace(body_replace)
print(f"The catagories in 'body' are : ",list(df['body'].unique()))


# In[17]:


#'transmission' category
df['transmission'] = df['transmission'].str.strip().str.lower() 
df['transmission'] = df['transmission'].replace({'sedan': np.nan})

#'state' category
df['state'] = df['state'].str.strip().str.upper() 
df.loc[~df['state'].str.match(r'^[A-Z]{2}$'), 'state'] = np.nan

#'color' category
df['color'] = df['color'].replace("—", np.nan)
df['color'] = df['color'].where(df['color'].str.isalpha(), np.nan)  

#'interior' category
df['interior'] = df['interior'].replace("—", np.nan)

# Fill NaN with mode
for col in ['transmission', 'state', 'color', 'interior']:
    if df[col].isna().sum() > 0:
        mode_val = df[col].mode()[0]  
        df[col] = df[col].fillna(mode_val)

for i in df[['transmission', 'state', 'color', 'interior']]:
    print(f"The catagories in '{i}' are : ",list(df[i].unique()))


# In[18]:


#Export cleaned dataset
df.to_csv('car_prices_cleaned.csv', index=False)


# In[19]:


# Exploratory Data Analysis (EDA)
# Univariate Analysis
#Histogram
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


# In[20]:


#Countplot
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


# In[21]:


# Multivariate Analysis
#Check Multicollinearity of Numerical Features
#Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="crest")
plt.title("Multivariate Analysis (Correlation of Numerical Features)")
plt.savefig("03_Multivariate_Analysis_(Correlation_of_Numerical_Features).jpg", bbox_inches='tight')
plt.show()


# In[22]:


#Brand vs Selling Price
if 'make' in df.columns:
    plt.figure(figsize=(14,7))
    sns.boxplot(x='make', y='sellingprice', data=df, palette='flare')
    plt.xticks(rotation=90)
    plt.title("Brand vs Selling Price")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("04_Brand_vs_Selling_Price_(boxplot).jpg", bbox_inches='tight')
    plt.show()


# In[23]:


#Odometer vs Selling Price
if 'odometer' in df.columns:
    plt.figure(figsize=(8,5))
    sns.scatterplot(x='odometer', y='sellingprice', data=df, hue='body', palette='magma')
    plt.title("Odometer vs Selling Price")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("05_Odometer_vs_Selling_Price_(scatterplot).jpg", bbox_inches='tight')
    plt.show()


# In[24]:


#Condition vs MMR
if 'condition' in df.columns:
    plt.figure(figsize=(8,5))
    sns.lineplot(x='condition', y='mmr', data=df)
    plt.title("Cars' Conditions vs Manheim Market Report (MMR)")
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig("06_Condition_vs_MMR.jpg", bbox_inches='tight')
    plt.show()


# In[25]:


# Data Visualization
#Trend

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


# In[26]:


#Costliest Brand and Costliest Car
top_model = (df['make'] + ' - ' + df['model']).value_counts().head(10).reset_index()
top_model.columns = ['BrandModel', 'Quantity']

max_val = top_model['Quantity'].max()

fig = px.bar(
    top_model,
    x='BrandModel',
    y='Quantity',
    title='Top 10 Most Popular Cars (Brand-Model)',
    text='Quantity'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    yaxis=dict(range=[0, max_val * 1.2]),
    xaxis_tickangle=-45
)

fig.show()
fig.write_html("08_Top_10_Most_Popular_Cars.html")


# In[27]:


# Combine brand and model
df['BrandModel'] = df['make'] + ' - ' + df['model']

# Group by BrandModel and sum the selling price
top_model_revenue = (
    df.groupby('BrandModel')['sellingprice']
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)
top_model_revenue['TotalRevenueMillions$'] = (top_model_revenue['sellingprice'] / 1e6).round(2)

# Get max revenue value to set y-axis limit
max_val = top_model_revenue['TotalRevenueMillions$'].max()

# Create the bar chart
fig = px.bar(
    top_model_revenue,
    x='BrandModel',
    y='TotalRevenueMillions$',
    title='Top 10 Cars by Total Sales Revenue (in Millions)',
    text='TotalRevenueMillions$'
)

fig.update_traces(textposition='outside')

fig.update_layout(
    yaxis=dict(range=[0, max_val * 1.2]),  # 20% headroom
    yaxis_title='Revenue (Millions $)',
    xaxis_tickangle=-45,
    xaxis=dict(categoryorder='total descending')
)

fig.show()
fig.write_html("09_Top_10_Cars_by_Total_Sales_Revenue.html")


# In[28]:


#Treemap
model_revenue = (
    df.groupby(['make', 'model'])['sellingprice']
    .sum()
    .reset_index()
    .sort_values(by='sellingprice', ascending=False)
)

fig = px.treemap(
    model_revenue.head(50),  # Limit to top 50 for clarity
    path=['make', 'model'],
    values='sellingprice',
    title='Top Cars Models by Total Revenue'
)
fig.show()
fig.write_html("10_Top_ Cars_Models_by_Total_Revenue_(treemap).html")


# In[29]:


#scatter
fig = px.scatter(
    df,
    x='year',
    y='sellingprice',
    color='make',  
    title='Car Selling Price vs Manufacturing Year by Brand',
    opacity=0.6,
    height=600
)

fig.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title='Selling Price ($)',
    xaxis_title='Year',
    legend_title='Brand',
    margin=dict(l=60, r=20, t=60, b=60)
)

fig.show()
fig.write_html("11_Cars_Selling_Price_vs_Manufacturing_Year_by_Brand_(scatter).html")


# In[30]:


state_revenue = df.groupby('state')['sellingprice'].sum().reset_index()

fig = px.choropleth(
    state_revenue,
    locations='state',
    locationmode="USA-states",
    color='sellingprice',
    color_continuous_scale="cividis",
    scope="usa",
    title="Total Sales Revenue by State"
)
fig.show()
fig.write_html("12_Total_Sales_Revenue_by_State.html")


# In[31]:


# Data Modeling
#target
y = df['sellingprice']

#predictor features
X = df[['year', 'condition', 'odometer', 'mmr']]

#split: train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Linear Regression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#prediction
y_pred = linreg.predict(X_test)

#evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("=== Linear Regression ===")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

#coef feature
coef = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': linreg.coef_
}).sort_values(by='Coefficient', ascending=False)
print("\nCoefficient Linear Regression:")
print(coef)

print("Intercept:", linreg.intercept_)

#Save model
joblib.dump(linreg, "linear_regression_model_car.pkl")

#Load model
loaded_model = joblib.load("linear_regression_model_car.pkl")


# In[32]:


#Scatterplot selling price vs prediction price
plt.figure(figsize=(7,6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, hue=(y_pred - y_test), palette="hsv", legend=False)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  #ideal line
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual Selling Price vs Predicted Selling Price")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()


# In[33]:


# Distribution Plot Error (Residuals)
errors = y_test - y_pred
plt.figure(figsize=(7,5))
sns.histplot(errors, bins=40, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.xlabel("Error (y_true - y_pred)")
plt.title("Error (Residuals) Distribution")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()


# In[34]:


#Error Absolute vs Selling Price
abs_errors = abs(errors)
plt.figure(figsize=(7,5))
sns.scatterplot(x=y_test, y=abs_errors, alpha=0.5)
plt.xlabel("Selling Price")
plt.ylabel("Absolute Error")
plt.title("Absolute Error to Selling Price")
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.show()


# In[ ]:




