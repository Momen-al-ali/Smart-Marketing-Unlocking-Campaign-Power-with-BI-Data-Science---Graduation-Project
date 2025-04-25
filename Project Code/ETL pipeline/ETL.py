import pandas as pd
import pyodbc

# Extract: Load CSV file into a DataFrame
def extract_data(file_path):
    return pd.read_csv(file_path)

# Transform: Apply transformations to the dataset
def transform_data(dataset):
    # 1. Split "Target_Audience" into "Gender" and "Age_Segment"
    dataset[['Gender', 'Age_Segment']] = dataset['Target_Audience'].str.split(r'\s+', expand=True)
    
    # 2. Replace values in "Age_Segment"
    dataset['Age_Segment'] = dataset['Age_Segment'].replace({'18-24': '18-24', '25-34': '25-34', '35-44': '35-44', 'Ages': '45+'})
    
    # 3. Remove 'days' from "Duration" and convert to integer
    dataset['Duration'] = dataset['Duration'].str.replace('days', '').astype(int)
    
    # 4. Rename "Duration" column to "Duration_Days"
    dataset.rename(columns={'Duration': 'Duration_Days'}, inplace=True)
    
    # 5. Rename values in "Age_Segment"
    dataset['Age_Segment'] = dataset['Age_Segment'].replace({'18-24': 'Gen Z', '25-34': 'Millennials', '35-44': 'Gen X', '45+': 'Baby Boomers'})
    
    # 6. Remove '$' and ',' from "Acquisition_Cost" and convert to float
    dataset['Acquisition_Cost'] = dataset['Acquisition_Cost'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # 7. Create "Is_Successful" column
    dataset['ROI'] = pd.to_numeric(dataset['ROI'], errors='coerce')
    dataset['Engagement_Score'] = pd.to_numeric(dataset['Engagement_Score'], errors='coerce')
    mean_roi = dataset['ROI'].mean()
    dataset['Is_Successful'] = ((dataset['ROI'] > mean_roi) & (dataset['Engagement_Score'] > 7)).astype(int)
    
    return dataset

# Load: Save the transformed data to SQL Server
def load_data_to_sql(dataset, table_name, server_name, database_name):
    # Create connection to SQL Server
    connection = pyodbc.connect(
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server_name};"
        f"DATABASE={database_name};"
        "Trusted_Connection=yes;"
    )
    cursor = connection.cursor()
    
    # Create table if it doesn't exist
    columns = ", ".join(
        [f"[{col}] NVARCHAR(MAX)" if dataset[col].dtype == 'object' else f"[{col}] FLOAT" if dataset[col].dtype == 'float64' else f"[{col}] INT" for col in dataset.columns]
    )
    create_table_query = f"""
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')
    CREATE TABLE {table_name} ({columns})
    """
    cursor.execute(create_table_query)
    connection.commit()
    
    # Insert data into the table
    for _, row in dataset.iterrows():
        values = ", ".join(
            ["'{}'".format(str(value).replace("'", "''")) if isinstance(value, str) else str(value) for value in row]
        )
        insert_query = f"INSERT INTO {table_name} VALUES ({values})"
        cursor.execute(insert_query)
    connection.commit()
    cursor.close()
    connection.close()

# Main ETL pipeline function
def etl_pipeline(file_path, table_name, server_name, database_name):
    # Extract
    dataset = extract_data(file_path)
    
    # Transform
    transformed_dataset = transform_data(dataset)
    
    # Load
    load_data_to_sql(transformed_dataset, table_name, server_name, database_name)

# Run the ETL pipeline
if __name__ == "__main__":
    file_path = 'D:\Gradution Project V2\Data\Row Data\marketing_campaign_dataset.csv'  # Replace with your CSV file path
    table_name = 'MarketingCampaignData'  # Replace with your desired table name
    server_name = r'LAPTOP-DQM4976N\SQLEXPRESS'  # SQL Server instance name
    database_name = 'MarketingDB'  # Replace with the actual database name
    
    etl_pipeline(file_path, table_name, server_name, database_name)