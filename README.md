# LastMinute

A quickly duct taped API wrapper around a ML neural network. For use as a web
services replacement.

## Requests

### /nn-submit

Request Body (JSON):
```
{
  "age": 60,
  "high_risk_exposure_occupation": false,
  "high_risk_interactions": false,
  "diabetes": true,
  "chd": false,
  "htn": false,
  "cancer": false,
  "asthma": true,
  "copd": false,
  "autoimmune_dis": true,
  "smoker": true,
  "temperature": 39,
  "pulse": 90,
  "labored_respiration": true,
  "cough": false,
  "fever": true,
  "sob": false,
  "diarrhea": false,
  "fatigue": false,
  "headache": false,
  "loss_of_smell": false,
  "loss_of_taste": false,
  "runny_nose": false,
  "muscle_sore": false,
  "sore_throat": false
}
```

Response Body (JSON):
```
{
    "prediction": 0.21778744459152222
}
```
