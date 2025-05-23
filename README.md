from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import io

# Generate synthetic retail data
data = {
    'CustomerID': range(1, 201),
    'Annual Income': np.random.randint(15000, 120000, 200),
    'Spending Score': np.random.randint(1, 100, 200)
}
df = pd.DataFrame(data)

# Data Preprocessing
# 1. Check for missing values
missing_summary = df.isnull().sum()

# 2. Standardize features
scaler = StandardScaler()
df[['Annual Income', 'Spending Score']] = scaler.fit_transform(df[['Annual Income', 'Spending Score']])

# 3. Show summary statistics after standardization
summary_stats = df[['Annual Income', 'Spending Score']].describe()

# 4. Visualize standardized features
df_melted = df.melt(id_vars='CustomerID', value_vars=['Annual Income', 'Spending Score'], var_name='Feature', value_name='Value')
plt.figure(figsize=(8, 5))
sns.histplot(data=df_melted, x='Value', hue='Feature', element='step', stat='density', common_norm=False)
plt.title('Distribution of Standardized Features')
plt.tight_layout()
img_buf = io.BytesIO()
plt.savefig(img_buf, format='png')
plt.close()
img_buf.seek(0)

# Prepare PDF content
pdf = FPDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 16)
pdf.cell(0, 10, 'Data Preprocessing for Customer Segmentation', ln=True, align='C')

pdf.set_font('Arial', '', 12)
pdf.ln(10)
pdf.multi_cell(0, 10, (
    'Data preprocessing is a crucial step in customer segmentation. It ensures that the data is clean, consistent, and ready for clustering. '
    'Here, we start by checking for missing values, then standardize the numerical features to have a mean of 0 and a standard deviation of 1.'
))

pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, '1. Checking for Missing Values', ln=True)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, (
    'We check each column for missing values. A summary is shown below.'
))
pdf.set_font('Courier', '', 10)
pdf.multi_cell(0, 7, missing_summary.to_string())

pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, '2. Standardizing Features', ln=True)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, (
    'Standardization rescales features so that they have a mean of 0 and a standard deviation of 1. This is important for clustering algorithms like K-means, which are sensitive to feature scales.'
))

pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, '3. Summary Statistics After Standardization', ln=True)
pdf.set_font('Courier', '', 8)
pdf.multi_cell(0, 5, summary_stats.to_string())

pdf.ln(5)
pdf.set_font('Arial', 'B', 12)
pdf.cell(0, 10, '4. Visualization of Standardized Features', ln=True)
pdf.set_font('Arial', '', 12)
pdf.multi_cell(0, 10, (
    'The histogram below shows the distribution of the standardized features. Both annual income and spending score are now centered around zero.'
))
pdf.image(img_buf, x=30, w=150)

filename = 'customer_segmentation_preprocessing.pdf'
pdf.output(filename)
print('PDF with data preprocessing section created as customer_segmentation_preprocessing.pdf')
