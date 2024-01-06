import streamlit as st
import pandas as pd
import numpy as np

# Assuming the necessary imports for your prediction pipeline
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

# Initialize your prediction pipeline
predict_pipeline = PredictPipeline()

# Creating the Streamlit app layout
st.title('Loan Application Form')

# Collecting user inputs
income = st.number_input('Income', min_value=0.0, format='%f')
debt_with_other_lenders = st.number_input('Debt with Other Lenders', min_value=0.0, format='%f')
credit_score = st.number_input('Credit Score', min_value=0.0, format='%f')
has_previous_defaults = st.checkbox('Has Previous Defaults with Other Lenders')
num_remittances_prev_12_mth = st.number_input('Number of Remittances in Previous 12 Months', min_value=0)
remittance_amt_prev_12_mth = st.number_input('Remittance Amount in Previous 12 Months', min_value=0.0, format='%f')
main_remittance_corridor = st.selectbox('Main Remittance Corridor', ['AE_IN', 'AE_PK', 'AE_PH'])
opened_campaign_1 = st.checkbox('Opened Campaign 1')
opened_campaign_2 = st.checkbox('Opened Campaign 2')
opened_campaign_3 = st.checkbox('Opened Campaign 3')
opened_campaign_4 = st.checkbox('Opened Campaign 4')
tenure_years = st.number_input('Tenure Years', min_value=0)

# Prediction button
if st.button('Predict'):
    # Create data instance
    data = CustomData(
        income=income,
        debt_with_other_lenders=debt_with_other_lenders,
        credit_score=credit_score,
        has_previous_defaults_other_lenders=has_previous_defaults,
        num_remittances_prev_12_mth=num_remittances_prev_12_mth,
        remittance_amt_prev_12_mth=remittance_amt_prev_12_mth,
        main_remittance_corridor=main_remittance_corridor,
        opened_campaign_1=opened_campaign_1,
        opened_campaign_2=opened_campaign_2,
        opened_campaign_3=opened_campaign_3,
        opened_campaign_4=opened_campaign_4,
        tenure_years=tenure_years
    )

    pred_df = data.get_data_as_data_frame()
    results = predict_pipeline.predict(pred_df)
    st.success(f'The prediction is {results[0]}')

# Run the Streamlit app with: streamlit run streamlit_app.py
