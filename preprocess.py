import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Load the dataset
df = pd.read_csv("./datasets/FastFoodNutritionMenuV2.csv")
df.columns = [name.replace('\n', " ") for name in df.columns]
print(df.info())
print(df.isnull().sum())

# df = df.dropna()
# print(df.info())


company_counts = df['Company'].value_counts()
colors = {
    "McDonald’s": '#ffd166',   
    "KFC": '#f78c6b',         
    "Burger King": '#ef476f', 
    "Taco Bell": '#118ab2',   
    "Wendy’s": '#073b4c',    
    "Pizza Hut": '#06d6a0'    
}

color_list = [colors[comp] for comp in company_counts.index]

fig, axs = plt.subplots(1, 2, figsize=(11, 5.5))
bars = axs[0].bar(company_counts.index, company_counts.values, color=color_list)
axs[0].set_title("Company Frequency Distribution (Histogram)")
axs[0].set_xlabel("Companies")
axs[0].set_ylabel("Frequency Count")
axs[0].tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    axs[0].text(bar.get_x() + bar.get_width()/2, height, f'{int(height)}',
                ha='center', va='bottom', fontsize=10, color='black')
legend_handles = [plt.Rectangle((0, 0), 1, 1, color=colors[comp]) for comp in company_counts.index]
axs[0].legend(legend_handles, company_counts.index, title="Company", fontsize=10)

wedges, texts, autotexts = axs[1].pie(company_counts.values,
           labels=company_counts.index,    # 如果想在饼图上显示标签，可保留该项
           autopct='%1.1f%%',
           startangle=90,
           colors=color_list,
           radius=1.1,
           wedgeprops={'edgecolor': 'black'})

axs[1].set_title("Company Frequency Distribution (Pie Chart)")
# axs[1].legend(wedges, company_counts.index, title="Company", bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Company_Frequency_Distribution.png")


columns_to_plot = [
    "Calories",
    "Calories from Fat",
    "Total Fat (g)",
    "Saturated Fat (g)",
    "Trans Fat (g)",
    "Cholesterol (mg)",
    "Sodium  (mg)",
    "Carbs (g)",
    "Fiber (g)",
    "Sugars (g)",
    "Protein (g)",
    "Weight Watchers Pnts"
]


for col in columns_to_plot:
    plt.figure(figsize=(6, 5))
    sns.histplot(pd.to_numeric(df[col], errors='coerce'), kde=True, bins=30)  
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{col}_distribution.png")

for col in columns_to_plot:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

plt.figure(figsize=(10, 8))
sns.heatmap(df[columns_to_plot].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Nutritional Features and Weight Watchers Pnts")
plt.tight_layout()
plt.savefig("Heatmap.png")