import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")

            # Preprocess the features using the loaded preprocessor
            data_scaled = preprocessor.transform(features)

            # Predict using the loaded model
            preds = model.predict(data_scaled)

            # Return predictions
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, income, debt_with_other_lenders, credit_score, 
                 has_previous_defaults_other_lenders, num_remittances_prev_12_mth, 
                 remittance_amt_prev_12_mth, main_remittance_corridor, 
                 opened_campaign_1, opened_campaign_2, opened_campaign_3, 
                 opened_campaign_4, tenure_years):

        self.income = income
        self.debt_with_other_lenders = debt_with_other_lenders
        self.credit_score = credit_score
        self.has_previous_defaults_other_lenders = has_previous_defaults_other_lenders
        self.num_remittances_prev_12_mth = num_remittances_prev_12_mth
        self.remittance_amt_prev_12_mth = remittance_amt_prev_12_mth
        self.main_remittance_corridor = main_remittance_corridor
        self.opened_campaign_1 = opened_campaign_1
        self.opened_campaign_2 = opened_campaign_2
        self.opened_campaign_3 = opened_campaign_3
        self.opened_campaign_4 = opened_campaign_4
        self.tenure_years = tenure_years

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "income": [self.income],
                "debt_with_other_lenders": [self.debt_with_other_lenders],
                "credit_score": [self.credit_score],
                "has_previous_defaults_other_lenders": [self.has_previous_defaults_other_lenders],
                "num_remittances_prev_12_mth": [self.num_remittances_prev_12_mth],
                "remittance_amt_prev_12_mth": [self.remittance_amt_prev_12_mth],
                "main_remittance_corridor": [self.main_remittance_corridor],
                "opened_campaign_1": [self.opened_campaign_1],
                "opened_campaign_2": [self.opened_campaign_2],
                "opened_campaign_3": [self.opened_campaign_3],
                "opened_campaign_4": [self.opened_campaign_4],
                "tenure_years": [self.tenure_years]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
