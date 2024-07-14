# Stock Price Prediction Using Machine Learning

## Introduction

This project aims to predict stock prices using various machine learning models, leveraging both basic and advanced techniques. The project also predicts future stock prices for the next 30 days, which is the main focus. By analyzing historical stock data and incorporating news sentiment analysis, this project provides a comprehensive approach to stock price prediction.
The Project helps in a clear understanding of how different prediction models perform in Stock Prediction. In most of the cases, Linear Regression and LSTM models outstand the performance by other models. Here are potential reasons for their relative strengths-
Linear Regression: 
Linear regression is straightforward to understand and implement. Its coefficients can provide insights into the relationship between features and the target variable (stock price).
For datasets with linear relationships between features and the target, linear regression can be computationally efficient.
It's often used as a baseline to compare against more complex models. Sometimes, a simple model can outperform complex ones when the underlying patterns are indeed linear.
LSTM:
Handling Time-Series Data: LSTM is specifically designed to handle sequential data, which is the nature of stock prices. It excels at capturing long-term dependencies and patterns that other models might struggle with.
Nonlinear Relationships: Stock price movements often exhibit complex, nonlinear patterns. LSTM's ability to learn complex patterns can be advantageous in such cases.
Feature Learning: LSTM can automatically learn relevant features from raw data, reducing the need for extensive feature engineering.
while Linear Regression and LSTM have shown potential in stock price prediction, their success depends on various factors. It's crucial to experiment with different models, evaluate their performance rigorously, and consider the specific characteristics of the dataset and prediction task.

## Data Collection Using yFinance

The project uses the yFinance library to collect historical stock data. yFinance is a powerful library that allows easy access to Yahoo Finance data, including stock prices, trading volume, and financial indicators. This data serves as the foundation for training and testing the prediction models.

## Importance of News Sentiment Analysis

News sentiment analysis helps in understanding the general sentiment and public emotion towards a company. By analyzing news articles, we can gauge whether the sentiment is positive, negative, or neutral, providing valuable insights into the company's reputation. Although the sentiment analysis is not included in the prediction models, it helps provide a holistic view of the company's standing in the market.

## Machine Learning Models

### Linear Regression
Linear regression is a basic prediction model that assumes a linear relationship between the input features and the target variable. It is simple yet effective for predicting stock prices based on historical data.

### Support Vector Machine (SVM)
SVM is a robust machine learning model that aims to find the hyperplane that best separates the data into different classes. In the context of stock price prediction, SVM helps in capturing complex relationships between the features and the target variable.

### Random Forest
Random Forest is an ensemble learning method that combines multiple decision trees to improve prediction accuracy. It is highly effective in handling large datasets and capturing intricate patterns in the data.

### Recurrent Neural Network (RNN)
RNN is a type of neural network designed for sequential data. It is particularly useful for time series prediction, as it can retain information from previous inputs, making it ideal for stock price prediction.

### Long Short-Term Memory (LSTM)
LSTM is a special type of RNN that can learn long-term dependencies in the data. It effectively mitigates the vanishing gradient problem, making it highly suitable for time series prediction tasks like stock price prediction.

## Sliding/Rolling Window Algorithm

The sliding or rolling window algorithm is used to predict future stock prices. This approach involves using a fixed-size window of past data to predict the next value. For example, if we have a window size of 100, we use the past 100 days of stock prices to predict the price on the 101st day. This process is repeated iteratively to predict stock prices for the next 30 days.

## Project Structure

The project is organized into six main files:

1. **NewsArticleSentimentAnalysis.ipynb**: Performs news sentiment analysis about the company, providing insights into its reputation and public emotion.

2. **DataCollection-preprocessing.ipynb**: Contains functions to collect historical data from yFinance and preprocess it for model training.

3. **BasicPredictionModels.ipynb**: Includes functions to train and visualize various basic prediction models such as Linear Regression, SVM, and Random Forest.

4. **NeuralNetworkModels.ipynb**: Contains functions to train and visualize neural network models like RNN and LSTM.

5. **ModelComparison.ipynb**: Compares the performance of all the prediction models used and returns the overall best predictor.

6. **StockPricePrediction.ipynb**: Acts as the main file for the project. It displays all the outputs and contains functions to predict stock prices for the next 30 days using the overall best predictor.

### Advantages of Maintaining Separate Files

- **Modularity**: Each file has a specific purpose, making the code easier to manage and maintain.
- **Reusability**: Functions in individual files can be reused across different parts of the project.
- **Scalability**: The project can be easily scaled by adding new models or data sources without affecting the existing structure.
- **Collaboration**: Multiple team members can work on different parts of the project simultaneously without conflicts.

## Optional: News API Generation

To perform news sentiment analysis, you need a News API key. Follow these steps to generate one:

1. Sign up at [News API](https://newsapi.org/register) to create an account.
2. After signing in, navigate to the API keys section to obtain your unique API key.
3. Replace the placeholder API key in the `NewsArticleSentimentAnalysis.ipynb` file with your own.

**Note**: News sentiment analysis is optional and is not included in the prediction models. It is intended to provide additional insights into the company's reputation.

---

With this comprehensive setup, you can predict stock prices using various machine learning models, visualize their performance, and forecast future prices with a sliding window approach. The modular structure ensures ease of maintenance and scalability, allowing for continuous improvements and updates.
