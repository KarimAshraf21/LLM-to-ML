from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from parser import CustomerData
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import joblib
import pandas as pd


def predict_churn(input_text):

    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    chat = ChatGroq(temperature=0, model_name="gemma2-9b-it")
    parser = JsonOutputParser(pydantic_object=CustomerData)
    prompt = PromptTemplate(
        template="""
        Convert the following customer description into structured JSON. If a variable is not mentioned, assume it is 0.
        {format_instructions}
        Description: {input}
        """,
        input_variables=["input"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    chain = prompt | chat | parser
    variables = chain.invoke(input=input_text)
    model = joblib.load("resources/churn_prediction_model.joblib")
    variables = pd.DataFrame([variables])
    predicted = model.predict(variables.values.reshape(1, -1))
    # print(f"Predicted Churn: {predicted[0]}")
    prompt = """
    You are a helpful assistant. Turn the following churn prediction into a natural language explanation for a customer service agent. Don't yap about it.

    Prediction: {prediction}

    Your response should clearly state if the customer is likely to churn or not. The input prediction is a binary variable whether the customer will
    churn or not
    """
    prediction_prompt = PromptTemplate(
        template= prompt,
        input_variables=["prediction"],
    )
    prediction_chain = prediction_prompt | chat
    # print(prediction_chain.invoke(input=predicted[0]).content)
    return prediction_chain.invoke(input=predicted[0]).content