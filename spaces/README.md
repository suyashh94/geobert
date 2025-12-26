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
---

# GeoBERT - NYC Address Geocoder

Predict geographic coordinates for New York City addresses using a fine-tuned BERT model.

## Model Details

- **Architecture:** Tiny BERT (`google/bert_uncased_L-2_H-128_A-2`) + Regression Head
- **Parameters:** ~4.4M total
- **Training Data:** ~1M NYC address points from NYC Open Data
- **Output:** Latitude and Longitude coordinates

## Usage

1. Enter an NYC address in the text box
2. Click "Geocode" or press Enter
3. View the predicted coordinates and interactive map

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
