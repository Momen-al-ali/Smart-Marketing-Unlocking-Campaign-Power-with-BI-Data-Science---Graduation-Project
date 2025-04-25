
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

class MLPipeline:
    def __init__(self, dataset):
        self.dataset = dataset

    def data_transformation(self):
        # Step 1: Split "Target_Audience" into "Gender" and "Age_Segment"
        self.dataset[['Gender', 'Age_Segment']] = self.dataset['Target_Audience'].str.split(r'\s+', expand=True)

        # Step 2: Replace values in "Age_Segment"
        self.dataset['Age_Segment'] = self.dataset['Age_Segment'].replace(
            {'18-24': '18-24', '25-34': '25-34', '35-44': '35-44', 'Ages': '45+'})

        # Step 3: Modify "Duration" column
        self.dataset['Duration'] = self.dataset['Duration'].str.replace('days', '').astype(int)

        # Step 4: Rename "Duration" column
        self.dataset.rename(columns={'Duration': 'Duration_Days'}, inplace=True)

        # Step 5: Rename values in "Age_Segment"
        self.dataset['Age_Segment'] = self.dataset['Age_Segment'].replace(
            {'18-24': 'Gen Z', '25-34': 'Millennials', '35-44': 'Gen X', 'Ages': 'Baby Boomers'})

        # Step 6: Modify "Acquisition_Cost" column
        self.dataset['Acquisition_Cost'] = self.dataset['Acquisition_Cost'].str.replace('$', '').astype(str)
        self.dataset['Acquisition_Cost'] = self.dataset['Acquisition_Cost'].str.replace(',', '').astype(float)
        
        
        # Step 7: Create target variable "Successful_Campaign"
        self.dataset['ROI'] = pd.to_numeric(self.dataset['ROI'], errors='coerce')
        self.dataset['Engagement_Score'] = pd.to_numeric(self.dataset['Engagement_Score'], errors='coerce')
        mean_roi = self.dataset['ROI'].mean()
        self.dataset['Is_Successful'] = ((self.dataset['ROI'] > mean_roi) &
                                        (self.dataset['Engagement_Score'] > 7)).astype(int)

    def preprocess_and_train(self):
        # Define features and target
        X = self.dataset.drop(columns=['Is_Successful', 'Target_Audience'])
        y = self.dataset['Is_Successful']

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define preprocessing for numerical and categorical columns
        numerical_cols = ['Duration_Days', 'Conversion_Rate', 'Acquisition_Cost', 'ROI', 'Engagement_Score',
                        'Impressions', 'Clicks']
        categorical_cols = ['Gender', 'Age_Segment', 'Location', 'Channel_Used', 'Campaign_Type', 'Customer_Segment']

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])

        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression())
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Evaluate model
        y_pred = pipeline.predict(X_test)
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=5)
        print("\nCross-Validation Scores:", scores)
        print("Mean CV Score:", scores.mean())

        return pipeline


# Example usage:
dataset = pd.read_csv('D:\Gradution Project V2\Data\Row Data\marketing_campaign_dataset.csv')
ml_pipeline = MLPipeline(dataset)
ml_pipeline.data_transformation()
trained_model = ml_pipeline.preprocess_and_train()


# Save the trained pipeline to a file
joblib.dump(trained_model, "D:\Gradution Project V2\Data\Row Data\marketing_campaign_dataset.csv")
print("Trained ML pipeline saved successfully.")