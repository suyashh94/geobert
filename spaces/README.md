---
title: GeoBERT NYC Geocoder
emoji: 🗺️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
pinned: false
license: mit
tags:
  - geocoding
  - bert
  - nyc
  - maps
  - uncertainty
  - mdn
---

# GeoBERT - NYC Address Geocoder

Predict geographic coordinates for New York City addresses using fine-tuned BERT models.

## Model Types

### Regression Model
- **Architecture:** Tiny BERT + Regression Head
- **Output:** Direct latitude/longitude prediction
- **Use case:** Fast, deterministic predictions

### MDN Model (Mixture Density Network)
- **Architecture:** Tiny BERT + MDN Head (5 Gaussian mixtures)
- **Output:** Probability distribution over coordinates
- **Features:**
  - Uncertainty quantification (σ_lat, σ_lon)
  - Confidence intervals (50%, 80%, 95%)
  - Visual confidence circles on map

## Model Details

- **Base Model:** Tiny BERT (`google/bert_uncased_L-2_H-128_A-2`)
- **Parameters:** ~4.4M total
- **Training Data:** ~1M NYC address points from NYC Open Data

## Usage

1. Enter an NYC address in the text box
2. Select model type (Regression or MDN)
3. Click "Geocode" or press Enter
4. View the predicted coordinates and interactive map
   - For MDN: see confidence circles showing 50%, 80%, 95% intervals

## Example Addresses

- `350 5th Avenue, Manhattan, NY 10118` (Empire State Building)
- `1 World Trade Center, Manhattan, NY 10007`
- `200 Eastern Parkway, Brooklyn, NY 11238` (Brooklyn Museum)

## Limitations

- Only trained on NYC addresses - will not generalize to other cities
- Accuracy varies by borough and address complexity
- Best results with standard street addresses including borough and ZIP code

## Links

- [Model Repository](https://huggingface.co/suyash94/geobert-nyc)
- [NYC Open Data](https://data.cityofnewyork.us/)
