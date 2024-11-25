# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import re  # For regular expressions (text processing)
import matplotlib.pyplot as plt  # For plotting visualizations
import seaborn as sns  # For creating aesthetically pleasing statistical graphics
from sklearn.metrics import confusion_matrix  # To generate a confusion matrix
import numpy as np  # For numerical computations\
import os

# Load the dataset
data_path = r"CSV Files\customer_support_tickets.csv"
if os.path.exists(data_path):
    print("File found!")
else:
    print("File not found!")


df = pd.read_csv(data_path)  # Read the dataset into a DataFrame

# Display the dataset preview
print("Dataset Preview:")
print(df.head(), "\n")  # Display the first 5 rows for a quick overview

# Display column names
print("Column names in dataset:\n", df.columns, "\n")  # Print all column names

# Define the target column and text column
target_column = 'Ticket Type'  # Column containing the actual labels/categories
text_column = 'Ticket Description'  # Column containing text descriptions

# Print label distribution
print(f"Using target column '{target_column}'.\n")  # Print the target column
label_distribution = df[target_column].value_counts()  # Count the frequency of each category
print("Label distribution in 'Ticket Type':")
print(label_distribution, "\n")  # Display the distribution of categories

# Plot label distribution
plt.figure(figsize=(10, 6))  # Set figure size
sns.barplot(x=label_distribution.index, y=label_distribution.values, palette="viridis")  # Create a bar plot
plt.title("Label Distribution in Ticket Type", fontsize=16)  # Add a title
plt.ylabel("Count", fontsize=12)  # Label y-axis
plt.xlabel("Ticket Type", fontsize=12)  # Label x-axis
plt.xticks(rotation=45, fontsize=10)  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout
plt.show()  # Display the plot

# Preprocess text
def preprocess_text(text):
    """Preprocesses text by converting to lowercase, removing punctuation, and handling NaN values."""
    if pd.isna(text):  # Check for NaN values
        return ""  # Return an empty string for NaN
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()  # Remove leading/trailing spaces

# Apply text preprocessing
df[text_column] = df[text_column].apply(preprocess_text)

# Rule-based classification using if-elif-else
def classify_ticket(description):
    """Classifies a ticket description into predefined categories using if-elif-else statements."""
    description = description.lower()  # Convert to lowercase for consistency
    
    if 'refund' in description or 'return' in description or 'money back' in description:
        return 'Refund request'
    elif 'technical' in description or 'error' in description or 'not working' in description or 'crash' in description or 'bug' in description:
        return 'Technical issue'
    elif 'cancel' in description or 'termination' in description or 'close account' in description:
        return 'Cancellation request'
    elif 'price' in description or 'spec' in description or 'details' in description or 'buy' in description or 'purchase' in description:
        return 'Product inquiry'
    elif 'bill' in description or 'invoice' in description or 'charge' in description or 'payment' in description or 'account balance' in description:
        return 'Billing inquiry'
    elif 'ship' in description or 'delivery' in description or 'track' in description or 'shipping' in description or 'dispatch' in description:
        return 'Shipping'
    elif 'question' in description or 'help' in description or 'general' in description or 'information' in description or 'assistance' in description:
        return 'General Inquiry'
    else:
        return 'General Inquiry'  # Default category if no keywords match

# Apply classification to the dataset
df['Predicted_Category'] = df[text_column].apply(classify_ticket)

# Display predicted categories for the first few rows
print("Predicted categories:")
print(df[['Predicted_Category']].head(), "\n")  # Display the first 5 predictions

# Generate a confusion matrix
conf_matrix = confusion_matrix(df[target_column], df['Predicted_Category'], labels=df[target_column].unique())

# Plot Confusion Matrix Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap="Blues",
            xticklabels=df[target_column].unique(),
            yticklabels=df[target_column].unique())
plt.title("Confusion Matrix Heatmap", fontsize=16)
plt.ylabel("True Labels", fontsize=12)
plt.xlabel("Predicted Labels", fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()

# Visualize predicted categories (Pie Chart)
predicted_distribution = df['Predicted_Category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(predicted_distribution, labels=predicted_distribution.index, autopct='%1.1f%%',
        colors=sns.color_palette("pastel", len(predicted_distribution)))
plt.title("Distribution of Predicted Categories", fontsize=16)
plt.tight_layout()
plt.show()

# Simulate predicting a new ticket
new_ticket = "I have an issue with my invoice and charges"  # Define a new ticket description
new_ticket_processed = preprocess_text(new_ticket)  # Preprocess the ticket
predicted_category = classify_ticket(new_ticket_processed)  # Predict the category
print("Predicted category for new ticket:", predicted_category)  # Print the prediction
