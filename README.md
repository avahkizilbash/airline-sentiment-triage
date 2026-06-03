# Airline Tweet Sentiment Triage

A multiclass sentiment classifier that helps airline customer service teams prioritize which tweets need a human response — and which don't.

## The Problem

When a flight is delayed or a bag goes missing, customers tweet. For an airline's social media team, the challenge isn't finding negative tweets — it's finding them fast enough to respond before frustration compounds. A team that has to read every tweet manually can't scale. One that just responds to everything loses the ability to prioritize.

This project builds a classifier to do the triage: flag tweets that likely need a human response, surface the genuinely upset customers, and let the team focus their attention where it matters.

## The Question

Can the *language* of a tweet alone — no metadata, no account history, just the words — predict sentiment well enough to support customer service triage?

## Approach

- **Dataset:** ~14,000 tweets directed at major U.S. airlines, labeled as negative, neutral, or positive
- **Features:** Raw tweet text only (airline name excluded — this makes the model more general)
- **Model:** TF-IDF vectorization + multinomial logistic regression, tuned via 5-fold cross-validated grid search over vocabulary size, n-gram range, regularization strength, and stop words
- **Evaluation:** Class-specific precision, recall, and F1 — not just accuracy — because the cost of errors is asymmetric

## Why Accuracy Alone Is Misleading Here

The majority-class baseline achieves **63% accuracy** by predicting every tweet as negative. That sounds reasonable until you realize it completely fails to identify neutral and positive tweets — meaning it would flag routine questions and compliments as complaints. A real triage tool needs to distinguish between them.

## Results

| Metric | Majority-Class Baseline | TF-IDF + Logistic Regression |
|---|---|---|
| Accuracy | 63% | **80%** |
| Negative Recall | 100%* | **91%** |
| Macro F1 | 0.26 | **0.73** |
| Weighted F1 | 0.48 | **0.79** |

*The baseline achieves perfect negative recall only because it labels everything as negative — it identifies no neutral or positive tweets at all.

The selected model correctly classified **1,672 of 1,835** negative tweets in the held-out test set.

## What the Model Actually Learned

Extracting logistic regression coefficients shows the model learned real airline pain points — not noise:

**Strongest predictors of negative sentiment:** "worst," "delayed," "hold," "cancelled," "stuck," "luggage," "hours"

**Strongest predictors of neutral:** "hi," "can," "possible" — consistent with informational questions rather than emotional reactions

This interpretability matters if the model is going to inform real customer service decisions.

## Where It Fails

Error analysis of misclassified negative tweets reveals two main failure modes: **sarcasm** and **short, contextually ambiguous messages**. A tweet like "oh great, another delay" reads neutrally on word-level features. This supports using the model as a decision-support tool for human triage — not as a fully automated response system.

## Recommendation

Deploy as a prioritization layer, not a replacement for human judgment. The model catches 91% of negative tweets and substantially reduces the noise of neutral and positive messages in the queue. The 9% it misses tend to be the hard cases (sarcasm, ambiguity) that human representatives should review anyway.

## Tools

Python · pandas · scikit-learn · TF-IDF · logistic regression · GridSearchCV · matplotlib
