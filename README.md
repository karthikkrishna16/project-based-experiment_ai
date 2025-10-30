### NAME: TH KARTHIK KRISHNA

### REGISTER NO: 2122223240067

### DATE: 30/10/2025

###                                Project Based Experiment
Objective:

To perform sentiment analysis on Facebook data and filter for messages, comments, or posts with negative feedback.
#Program:
```

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download NLTK resources 
nltk.download('vader_lexicon')

# Load the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Example Facebook data 
facebook_data = [
  "I love the new feature! It's amazing.",
  "The service was terrible. I'm very disappointed.",
  "Great job on the update!",
  "The product quality is poor. I won't be buying again.",
  
]

# Perform sentiment analysis and filter for negative feedback
negative_feedback = []

for message in facebook_data:
  sentiment_score = sia.polarity_scores(message)['compound']
  if sentiment_score < 0:  # Negative sentiment
      negative_feedback.append(message)

# Print the negative feedback
print("Negative Feedback:")
for feedback in negative_feedback:
  print(feedback)
```

# Output:

![image](https://github.com/swathi22003343/Project-Based-Experiment-AAI/assets/120440439/3cbeb41c-321b-41cb-904a-b47c124cfafb)

# Result:
A sentiment analysis project using Facebook data provides valuable learning experiences in data handling, text processing, sentiment analysis, and ethical considerations, while also honing communication, problem-solving, and project management skills. 
# project-based-experiment_ai
