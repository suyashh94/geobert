"""GeoBERT Gradio App - NYC Address Geocoding with Map Visualization."""

import os

import folium
import gradio as gr
from gradio_folium import Folium
from inferencer import Inferencer

# Constants
NYC_CENTER = [40.7128, -74.0060]
REPO_ID = os.environ.get("HF_MODEL_REPO", "YOUR_HF_USERNAME/geobert-nyc")

# Initialize inferencer (loads model on startup)
print(f"Loading GeoBERT model from {REPO_ID}...")
inferencer = Inferencer(repo_id=REPO_ID)
print("Model ready!")


def create_prediction_map(lat: float, lon: float, address: str) -> folium.Map:
    """Create a Folium map with the predicted location.

    :param lat: Predicted latitude.
    :param lon: Predicted longitude.
    :param address: Input address string.
    :return: Folium Map object.
    """
    # Create map centered on prediction
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    # Add marker for prediction
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>Predicted Location</b><br>{address}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
            max_width=300,
        ),
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

    # Add circle to show approximate area
    folium.Circle(
        location=[lat, lon],
        radius=100,  # 100 meter radius
        color="red",
        fill=True,
        fill_opacity=0.2,
    ).add_to(m)

    return m


def geocode_address(address: str) -> tuple[str, folium.Map]:
    """Geocode an NYC address and return coordinates with map.

    :param address: NYC address string.
    :return: Tuple of (coordinates text, Folium map).
    """
    if not address or not address.strip():
        # Return default NYC map if no address
        empty_map = folium.Map(location=NYC_CENTER, zoom_start=11, tiles="CartoDB positron")
        return "Please enter an NYC address", empty_map

    # Run inference
    lat, lon = inferencer.predict(address.strip())
    lat, lon = float(lat[0]), float(lon[0])

    # Format output
    coords_text = f"""### Predicted Coordinates

**Latitude:** {lat:.6f}
**Longitude:** {lon:.6f}

[Open in Google Maps](https://www.google.com/maps?q={lat},{lon})
"""

    # Create map
    pred_map = create_prediction_map(lat, lon, address)

    return coords_text, pred_map


# Example addresses for the interface
EXAMPLE_ADDRESSES = [
    ["350 5th Avenue, Manhattan, NY 10118"],  # Empire State Building
    ["1 World Trade Center, Manhattan, NY 10007"],  # One WTC
    ["200 Eastern Parkway, Brooklyn, NY 11238"],  # Brooklyn Museum
    ["123-01 Roosevelt Avenue, Queens, NY 11368"],  # Citi Field area
    ["1000 Richmond Terrace, Staten Island, NY 10301"],  # Staten Island Ferry
    ["161st Street, Bronx, NY 10451"],  # Yankee Stadium area
]


# Build Gradio interface
with gr.Blocks(
    title="GeoBERT - NYC Address Geocoder",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
# GeoBERT - NYC Address Geocoder

Enter a New York City address to predict its geographic coordinates using a fine-tuned BERT model.

**Model:** Tiny BERT trained on ~1M NYC address points from [NYC Open Data](https://data.cityofnewyork.us/).
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            address_input = gr.Textbox(
                label="NYC Address",
                placeholder="e.g., 350 5th Avenue, Manhattan, NY 10118",
                lines=2,
            )
            geocode_btn = gr.Button("Geocode", variant="primary")

            gr.Examples(
                examples=EXAMPLE_ADDRESSES,
                inputs=[address_input],
                label="Example Addresses",
            )

            coords_output = gr.Markdown(label="Predicted Coordinates")

        with gr.Column(scale=2):
            map_output = Folium(
                value=folium.Map(location=NYC_CENTER, zoom_start=11, tiles="CartoDB positron"),
                height=500,
                label="Prediction Map",
            )

    # Event handlers
    geocode_btn.click(
        fn=geocode_address,
        inputs=[address_input],
        outputs=[coords_output, map_output],
    )

    address_input.submit(
        fn=geocode_address,
        inputs=[address_input],
        outputs=[coords_output, map_output],
    )

    gr.Markdown(
        """
---
**About:** This model uses a tiny BERT (`google/bert_uncased_L-2_H-128_A-2`) with a regression head,
fine-tuned on NYC address data. The model predicts latitude/longitude coordinates directly from address text.

**Limitations:**
- Only trained on NYC addresses - may not work for other locations
- Accuracy varies by borough and address format
- Best results with full addresses including borough and ZIP code
        """
    )


if __name__ == "__main__":
    demo.launch()
