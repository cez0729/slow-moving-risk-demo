# Methodology

## Why temporal splits

Retail observations are ordered in time. A random split can place future behavior in training and make the score look unrealistically strong. The demo trains on an earlier period, chooses the operating threshold on a later validation period, and evaluates once on a later locked-style test period.

## Why first-event labels

Repeated daily labels can count one long zero-sales episode many times. The demo records only the first continuous 30-day zero-sales period with positive stock. A product exits the risk set after that event, and only alerts 1-30 days before the event are valid.

## Why event metrics

The business action is an alert for an independent event, not a positive prediction on every risk day. Event-level TP, FP, FN, Precision, Recall, and F1 reflect duplicate-alert suppression and the 30-day cooldown.

## Leakage control

Rolling features include the current row and historical rows only. Prediction-mode feature generation does not compute future event labels. A dedicated test mutates all observations after a prediction date and checks that features at that date are unchanged.

## Validation and test separation

The threshold search receives validation rows only. The locked-style test is evaluated with that fixed threshold and is never used for model selection or tuning.
