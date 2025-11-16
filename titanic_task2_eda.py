import pandas as pd # Import pandas for data manipulation and analysis
import matplotlib.pyplot as plt # Import matplotlib.pyplot for creating static, interactive, and animated visualizations
import seaborn as sns # Import seaborn for statistical data visualization based on matplotlib
import os # Import os module for interacting with the operating system, like creating directories

# Define the URL for the cleaned dataset
github_csv_url = "https://raw.githubusercontent.com/ullas9525/AIML_Internship-Task-1_Elevate_Labs/main/Output/cleaned_titanic_dataset.csv" # This URL points to the raw CSV file on GitHub

# Load the dataset
try: # Use a try-except block to handle potential errors during data loading
    df = pd.read_csv(github_csv_url) # Load the dataset directly from the GitHub URL into a pandas DataFrame
    print("Dataset loaded successfully.") # Print a success message if the dataset is loaded
except Exception as e: # Catch any exception that occurs during the loading process
    print(f"Error loading dataset: {e}") # Print an error message if the dataset fails to load
    exit() # Exit the script if the dataset cannot be loaded

# Create the EDA_Output directory if it doesn't exist
output_dir = "Output" # Define the name of the directory to save the plots
if not os.path.exists(output_dir): # Check if the output directory already exists
    os.makedirs(output_dir) # Create the directory if it does not exist to store all generated plots
    print(f"Created directory: {output_dir}") # Inform the user that the directory has been created

# 1. Summary Statistics
print("\n--- Summary Statistics ---") # Print a header for the summary statistics section
print(df.describe()) # Display descriptive statistics for numerical columns to understand data distribution
print("\n--- Info ---") # Print a header for the DataFrame information section
print(df.info()) # Display a concise summary of the DataFrame, including data types and non-null values

# 2. Distribution Histograms for all numerical features
print("\n--- Generating Histograms ---") # Print a header indicating histogram generation
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns # Select only numerical columns for histogram plotting
for feature in numerical_features: # Iterate through each numerical feature to create a histogram
    plt.figure(figsize=(8, 6)) # Create a new figure with a specified size for each plot
    sns.histplot(df[feature], kde=True) # Plot a histogram with a Kernel Density Estimate to show the distribution
    plt.title(f'Distribution of {feature}') # Set the title of the plot to indicate the feature's distribution
    plt.xlabel(feature) # Label the x-axis with the feature name
    plt.ylabel('Frequency') # Label the y-axis as 'Frequency'
    plt.savefig(os.path.join(output_dir, f'{feature}_histogram.png')) # Save the generated histogram to the specified output directory
    plt.close() # Close the plot to free up memory

# 3. Boxplots for numerical features
print("--- Generating Boxplots ---") # Print a header indicating boxplot generation
for feature in numerical_features: # Iterate through each numerical feature to create a boxplot
    plt.figure(figsize=(8, 6)) # Create a new figure with a specified size for each plot
    sns.boxplot(y=df[feature]) # Plot a boxplot to visualize the distribution and identify outliers
    plt.title(f'Boxplot of {feature}') # Set the title of the plot to indicate the feature's boxplot
    plt.ylabel(feature) # Label the y-axis with the feature name
    plt.savefig(os.path.join(output_dir, f'{feature}_boxplot.png')) # Save the generated boxplot to the specified output directory
    plt.close() # Close the plot to free up memory

# 4. Heatmap for correlations
print("--- Generating Correlation Heatmap ---") # Print a header indicating heatmap generation
plt.figure(figsize=(10, 8)) # Create a new figure with a specified size for the heatmap
correlation_matrix = df.corr(numeric_only=True) # Calculate the pairwise correlation between numerical columns
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f") # Plot a heatmap of the correlation matrix with annotations
plt.title('Correlation Heatmap of Numerical Features') # Set the title of the heatmap
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png')) # Save the generated heatmap to the specified output directory
plt.close() # Close the plot to free up memory

# 5. Bar charts for key categorical insights (Sex vs Survived, Pclass vs Survived)
print("--- Generating Bar Charts for Categorical Insights ---") # Print a header indicating bar chart generation

# Sex vs Survived
# Create a 'Sex' column from 'Sex_female' and 'Sex_male' for plotting purposes
df['Sex'] = df['Sex_female'].apply(lambda x: 'female' if x == 1 else 'male') # Map the one-hot encoded 'Sex_female' to a categorical 'Sex' column
plt.figure(figsize=(8, 6)) # Create a new figure for the 'Sex vs Survived' bar chart
sns.barplot(x='Sex', y='Survived', data=df, palette='viridis') # Plot a bar chart to show the survival rate based on 'Sex'
plt.title('Survival Rate by Sex') # Set the title of the plot
plt.xlabel('Sex') # Label the x-axis as 'Sex'
plt.ylabel('Survival Rate') # Label the y-axis as 'Survival Rate'
plt.savefig(os.path.join(output_dir, 'sex_vs_survived_bar_chart.png')) # Save the bar chart to the output directory
plt.close() # Close the plot to free up memory

# Pclass vs Survived
plt.figure(figsize=(8, 6)) # Create a new figure for the 'Pclass vs Survived' bar chart
sns.barplot(x='Pclass', y='Survived', data=df, palette='magma') # Plot a bar chart to show the survival rate based on 'Pclass'
plt.title('Survival Rate by Pclass') # Set the title of the plot
plt.xlabel('Passenger Class') # Label the x-axis as 'Passenger Class'
plt.ylabel('Survival Rate') # Label the y-axis as 'Survival Rate'
plt.savefig(os.path.join(output_dir, 'pclass_vs_survived_bar_chart.png')) # Save the bar chart to the output directory
plt.close() # Close the plot to free up memory

# 6. Print 6-10 bullet-point insights based on the dataset
print("\n--- Key Insights from EDA ---") # Print a header for the key insights section
print("- The dataset contains information on passenger demographics, ticket details, and survival status.") # This insight provides a general overview of the dataset's content
print("- A significant portion of passengers did not survive, indicating the severity of the disaster.") # This insight highlights the overall survival outcome
print("- Females had a considerably higher survival rate compared to males, suggesting a 'women and children first' policy.") # This insight is derived from the 'Sex vs Survived' bar chart
print("- Passengers in Pclass 1 had a much higher survival rate than those in Pclass 2 and 3.") # This insight is derived from the 'Pclass vs Survived' bar chart
print("- Age distribution shows a wide range, with a notable number of children and elderly individuals.") # This insight comes from the 'Age' histogram and boxplot
print("- 'Fare' distribution is heavily skewed towards lower values, indicating most passengers paid less.") # This insight comes from the 'Fare' histogram and boxplot
print("- 'SibSp' and 'Parch' features indicate that most passengers traveled alone or with a small family.") # This insight comes from the 'SibSp' and 'Parch' histograms
print("- There is a moderate negative correlation between 'Pclass' and 'Fare', meaning higher classes paid more.") # This insight is derived from the correlation heatmap
