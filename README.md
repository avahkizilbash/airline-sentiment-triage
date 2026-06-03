# Airline Tweet Sentiment Triage

Airlines get thousands of tweets a day, and most of the real work is figuring out which ones need a person. A delayed-flight rant and a "thanks for the upgrade!" land in the same feed, but only one of them is a reputation risk sitting on a clock. This project asks whether a model reading nothing but the text of a tweet can sort that feed well enough to tell a customer service team where to look first.

The short answer is yes, mostly. The more useful answer is in where it breaks.

## What I built

I trained a classifier on roughly 14,000 tweets aimed at major U.S. airlines, each labeled negative, neutral, or positive. It uses TF-IDF to turn raw tweet text into numeric features and a logistic regression to make the call, with hyperparameters picked by cross-validated grid search. I left the airline name out of the features on purpose, so the model learns what frustration sounds like rather than which carrier gets complained about most.

## What I found

Accuracy is the wrong number to lead with, and most of the analysis is about why. You can hit 63% accuracy just by labeling every tweet negative, since negative is the majority class. That model is worthless in practice: it flags every compliment and every routine question as a complaint, so it gives the team nothing to prioritize.

The number I actually cared about is recall on negative tweets, meaning the share of real complaints the model catches. The final model gets to 80% overall accuracy and catches 91% of negative tweets (1,672 of 1,835 in the held-out set) while still telling neutral and positive messages apart instead of lumping everything together.

| Metric | Always-negative baseline | This model |
|---|---|---|
| Accuracy | 63% | 80% |
| Negative recall | 100% (by cheating) | 91% |
| Macro F1 | 0.26 | 0.73 |
| Weighted F1 | 0.48 | 0.79 |

Reading the model's coefficients back out confirms it learned something real rather than gaming the labels. The terms that push hardest toward negative are the actual failure points of air travel: worst, delayed, cancelled, hold, stuck, luggage, hours. The neutral signal is the language of people asking questions (hi, can, possible) rather than people who are upset.

## Where it breaks, and why that matters

The misses are the interesting part. When the model gets a negative tweet wrong, it's almost always sarcasm ("oh perfect, another two-hour delay") or something short and ambiguous that needs context the words alone don't carry. That's the whole case for how you'd actually deploy this. It's good enough to rank a queue and put the likely complaints in front of a person first. It is not good enough to answer tweets on its own, because the ones it misreads are exactly the ones where a wrong automated reply would embarrass the brand.

So the recommendation is narrow on purpose: use it to triage the queue, not to run it. It reliably catches nine of every ten complaints and clears most of the noise, and a human still handles the hard tenth.

## Stack
Python, pandas, scikit-learn (TF-IDF, logistic regression, GridSearchCV), matplotlib.
