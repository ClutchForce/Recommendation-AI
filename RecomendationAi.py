
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_all_customer_names(file_path):
    data = pd.read_csv(file_path)
    unique_names = data['Customer Name'].unique()
    #sort names alphabetically
    unique_names.sort()
    return unique_names

def print_customer_names_in_columns(names, columns=5):
    for i, name in enumerate(names):
        print(f"{name:<15}", end="")  # Print name with padding for alignment
        if (i + 1) % columns == 0:  # After every 'columns' names, print a new line
            print()
    print()  # Print a final new line at the end

def load_and_preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    
    # Remove unnecessary columns
    data = data.drop(columns=['Unnamed: 4', 'Unnamed: 5'])

    # Convert customer names to lower case
    data['Customer Name'] = data['Customer Name'].str.lower()  
    
    # Pivot to create the user-item interaction matrix for sub-categories
    user_item_matrix = pd.pivot_table(data, index='Customer Name', columns='Sub Category', aggfunc='size', fill_value=0)
    
    return user_item_matrix

def calculate_item_similarity(user_item_matrix):
    # Calculate the cosine similarity between sub-categories
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)
    
    return item_similarity_df

def recommend_for_user_modified(user_name, user_item_matrix, item_similarity_matrix, top_n=5):
    # Modified recommendation function
    if user_name not in user_item_matrix.index:
        return f"User '{user_name}' not found in the dataset."

    user_purchases = user_item_matrix.loc[user_name]
    recommendations = {}

    for sub_category, quantity in user_purchases.items():
        similar_items = item_similarity_matrix[sub_category]
        for similar_item, similarity in similar_items.items():  # Changed from iteritems() to items()
            if similar_item != sub_category:  # Exclude the sub-category itself
                weighted_similarity = similarity * (1 + quantity)
                recommendations[similar_item] = recommendations.get(similar_item, 0) + weighted_similarity

    recommended_sub_categories = sorted(recommendations, key=recommendations.get, reverse=True)[:top_n]
    return recommended_sub_categories

# Main flow
file_path = 'grocery_sells.csv'  # Update this with the actual file path
user_item_matrix = load_and_preprocess_data(file_path)
item_similarity_df = calculate_item_similarity(user_item_matrix)
all_customer_names = get_all_customer_names(file_path)


try:
    print("List of all customer names:")
    print_customer_names_in_columns(all_customer_names, 5)
    while True:  # Infinite loop
        user_name = input("Enter a user name to get recommendations (or type 'exit' to stop): ")
        if user_name.lower() == 'exit':  # Convert input to lower case for case-insensitive comparison
            break
        recommended_sub_categories = recommend_for_user_modified(user_name.lower(), user_item_matrix, item_similarity_df)
        print(f"Recommended sub-categories for '{user_name}': {recommended_sub_categories}")
except KeyboardInterrupt:
    print("\nScript terminated by user.")
