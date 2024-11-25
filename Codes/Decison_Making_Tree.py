# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
import matplotlib.pyplot as plt  # For plotting visualizations
import seaborn as sns  # For creating visually appealing charts

# Load the dataset
data_path = r"C:\Users\khadi\OneDrive\Desktop\AI Assignment Dataset\customer_support_tickets.csv"  # Path to the dataset
df = pd.read_csv(data_path)  # Read the CSV file into a DataFrame

# Display dataset info
print("Dataset Preview:\n", df.head())  # Display the first 5 rows of the dataset for an overview
print("\nColumn names in dataset:\n", df.columns)  # Display the column names to understand the structure

# Define the target and text columns
# Use 'Ticket Type' as the target column if it exists; otherwise, use 'Predicted_Category'
target_column = 'Ticket Type' if 'Ticket Type' in df.columns else 'Predicted_Category'
text_column = 'Ticket Description'  # The column containing textual data for classification
print(f"\nUsing target column '{target_column}'.")  # Confirm the chosen target column

# Label distribution
print(f"\nLabel distribution in '{target_column}':\n{df[target_column].value_counts()}")  # Display the count of each category in the target column

# Text preprocessing function
def preprocess_text(text):
    import re  # Import regular expressions module for text cleaning
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove all digits
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with 1-2 characters
    text = re.sub(r'[^\w\s]', '', text)  # Remove all punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and strip leading/trailing spaces
    return text

# Apply preprocessing to the text column
df[text_column] = df[text_column].apply(preprocess_text)

# Rule-based AI logic for category prediction
def predict_category(description):
    description = description.lower()  # Ensure the input description is in lowercase
    # Define keyword-based rules for each category
    if 'payment' in description or 'invoice' in description:
        return 'Billing'
    elif 'login' in description or 'password' in description:
        return 'Technical Support'
    elif 'delivery' in description or 'shipping' in description:
        return 'Shipping'
    elif 'refund' in description or 'return' in description:
        return 'Returns'
    else:
        return 'General Inquiry'  # Default category if no specific keywords match

# Predict categories for the entire dataset
df['Predicted_Category'] = df[text_column].apply(predict_category)

# Show predictions for the first few rows
print("\nPredicted categories:\n", df[['Predicted_Category']].head())  # Display predicted categories for a quick check

# Evaluation: Calculate Actual and Predicted Counts
actual_counts = df[target_column].value_counts()  # Count the actual occurrences of each category
predicted_counts = pd.Series(df['Predicted_Category']).value_counts()  # Count the predicted occurrences of each category

# Ensure both series align on categories (including categories missing in one of them)
all_categories = actual_counts.index.union(predicted_counts.index)  # Combine categories from both series
actual_counts = actual_counts.reindex(all_categories, fill_value=0)  # Fill missing categories with 0 in actual counts
predicted_counts = predicted_counts.reindex(all_categories, fill_value=0)  # Fill missing categories with 0 in predicted counts

# Chart 1: Precision of Predicted Categories (Pie Chart)
plt.figure(figsize=(8, 8))  # Set the figure size for the pie chart
plt.pie(
    predicted_counts,  # Data for the pie chart
    labels=predicted_counts.index,  # Labels for each slice
    autopct='%1.1f%%',  # Display percentages with 1 decimal place
    startangle=90,  # Start the pie chart at 90 degrees
    colors=sns.color_palette("pastel"),  # Use a pastel color palette
    wedgeprops={'edgecolor': 'black'}  # Add a black edge to each slice for better visibility
)
plt.title('Predicted Category Distribution', fontsize=14)  # Add a title to the chart
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()  # Display the pie chart

# Chart 2: Actual vs. Predicted Counts (Bar Chart)
plt.figure(figsize=(10, 6))  # Set the figure size for the bar chart
bar_width = 0.35  # Width of each bar in the chart
index = range(len(all_categories))  # Index positions for the categories

# Plot actual counts as the first set of bars
plt.bar(index, actual_counts, bar_width, label='Actual Counts', color='skyblue')
# Plot predicted counts as the second set of bars, offset by the bar width
plt.bar([i + bar_width for i in index], predicted_counts, bar_width, label='Predicted Counts', color='salmon')

# Configure x-axis
plt.xticks([i + bar_width / 2 for i in index], all_categories, rotation=45)  # Center ticks and rotate labels
plt.xlabel('Categories')  # Label for x-axis
plt.ylabel('Counts')  # Label for y-axis
plt.title('Distribution of Actual vs Predicted Counts by Category')  # Add a title
plt.legend()  # Display a legend
plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()  # Display the bar chart

# Sample prediction
new_ticket = "I want to know about my last invoice"  # Example input for prediction
new_ticket_processed = preprocess_text(new_ticket)  # Preprocess the new ticket description
predicted_category = predict_category(new_ticket_processed)  # Predict the category
print("\nPredicted category for new ticket:", predicted_category)  # Display the predicted category
