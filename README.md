# Restaurant Review Sentiment Analysis

This project aims to perform sentiment analysis on restaurant reviews to classify them as positive, negative, or neutral. It uses the Yelp dataset, containing user reviews of various restaurants.

## Problem Description

In the era of online reviews, businesses rely on customer feedback to understand their performance and improve their services. Analyzing restaurant reviews can help business owners identify areas of improvement, and potential customers can use the information to make informed decisions. The objective of this project is to perform sentiment analysis on restaurant reviews to classify them as positive, negative, or neutral. We'll use the Yelp dataset, containing user reviews of various restaurants.

## Dataset

The dataset used in this project can be downloaded from the [Yelp Dataset](https://www.yelp.com/dataset) page. It's available in JSON format and contains information about businesses, user reviews, and other data. For this project, we'll focus on the `review.json` file, which contains the user reviews and ratings.

## Dependencies

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

You can install these dependencies using `pip`:
pip install pandas numpy scikit-learn seaborn matplotlib


## Usage

1. Download the Yelp dataset and place the `review.json` file in the project directory.

2. Adjust the number of rows to load from the dataset in the `load_data()` function based on your computing power.

3. Run the `sentiment_analysis.py` script to train and evaluate the sentiment analysis model.

## Results

The model's performance is evaluated using accuracy, confusion matrix, and classification report. A heatmap of the confusion matrix is also visualized using Seaborn and Matplotlib.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
