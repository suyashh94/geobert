"""GeoBERT Gradio App - NYC Address Geocoding with Map Visualization."""

import os

import folium
import gradio as gr
from gradio_folium import Folium
from inferencer import Inferencer

# Constants
NYC_CENTER = [40.7128, -74.0060]
REPO_ID = os.environ.get("HF_MODEL_REPO", "suyash94/geobert-nyc")
LOCAL_DIR = os.environ.get("LOCAL_CHECKPOINT_DIR", None)  # For local testing

# Confidence levels for MDN visualization
CONFIDENCE_LEVELS = {
    0.50: {"z_score": 0.6745, "color": "blue", "opacity": 0.4, "fill_opacity": 0.2},
    0.80: {"z_score": 1.2816, "color": "blue", "opacity": 0.3, "fill_opacity": 0.1},
    0.95: {"z_score": 1.9600, "color": "blue", "opacity": 0.2, "fill_opacity": 0.05},
}

# Degrees to meters conversion at NYC latitude
DEG_LAT_TO_M = 111320
DEG_LON_TO_M = 84400

# Initialize both inferencers (loads models on startup)
inferencers = {}


def get_inferencer(model_type: str) -> Inferencer:
    """Get or create an inferencer for the given model type."""
    if model_type not in inferencers:
        if LOCAL_DIR:
            print(f"Loading GeoBERT {model_type} model from local: {LOCAL_DIR}")
            inferencers[model_type] = Inferencer(local_dir=LOCAL_DIR, model_type=model_type)
        else:
            print(f"Loading GeoBERT {model_type} model from HuggingFace Hub: {REPO_ID}")
            inferencers[model_type] = Inferencer(repo_id=REPO_ID, model_type=model_type)
    return inferencers[model_type]


# Pre-load regression model on startup
print("Pre-loading regression model...")
get_inferencer("regression")
print("Model ready!")


def create_regression_map(lat: float, lon: float, address: str) -> folium.Map:
    """Create a Folium map for regression model prediction.

    :param lat: Predicted latitude.
    :param lon: Predicted longitude.
    :param address: Input address string.
    :return: Folium Map object.
    """
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>Predicted Location</b><br>{address}<br>Lat: {lat:.6f}<br>Lon: {lon:.6f}",
            max_width=300,
        ),
        icon=folium.Icon(color="red", icon="map-marker", prefix="fa"),
    ).add_to(m)

    folium.Circle(
        location=[lat, lon],
        radius=100,
        color="red",
        fill=True,
        fill_opacity=0.2,
    ).add_to(m)

    return m


def create_mdn_map(
    lat: float,
    lon: float,
    sigma_lat: float,
    sigma_lon: float,
    address: str,
) -> folium.Map:
    """Create a Folium map for MDN model prediction with confidence circles.

    :param lat: Predicted latitude.
    :param lon: Predicted longitude.
    :param sigma_lat: Standard deviation of latitude (degrees).
    :param sigma_lon: Standard deviation of longitude (degrees).
    :param address: Input address string.
    :return: Folium Map object with confidence circles.
    """
    m = folium.Map(
        location=[lat, lon],
        zoom_start=15,
        tiles="CartoDB positron",
    )

    # Convert sigma to meters and use max for circular representation
    sigma_lat_m = sigma_lat * DEG_LAT_TO_M
    sigma_lon_m = sigma_lon * DEG_LON_TO_M
    sigma_max_m = max(sigma_lat_m, sigma_lon_m)

    # Draw confidence circles (outer to inner for proper layering)
    for conf_level in sorted(CONFIDENCE_LEVELS.keys(), reverse=True):
        style = CONFIDENCE_LEVELS[conf_level]
        radius_m = style["z_score"] * sigma_max_m

        folium.Circle(
            location=[lat, lon],
            radius=radius_m,
            color=style["color"],
            weight=1,
            opacity=style["opacity"],
            fill=True,
            fill_color=style["color"],
            fill_opacity=style["fill_opacity"],
            popup=f"{int(conf_level * 100)}% CI: {radius_m:.0f}m radius",
        ).add_to(m)

    # Add center marker
    folium.Marker(
        location=[lat, lon],
        popup=folium.Popup(
            f"<b>Predicted Location</b><br>{address}<br>"
            f"Lat: {lat:.6f}<br>Lon: {lon:.6f}<br>"
            f"<hr>"
            f"<b>Uncertainty:</b><br>"
            f"σ_lat: {sigma_lat:.6f}°<br>"
            f"σ_lon: {sigma_lon:.6f}°<br>"
            f"σ_max: {sigma_max_m:.0f}m",
            max_width=300,
        ),
        icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
    ).add_to(m)

    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white;
                padding: 10px; border: 2px solid gray; border-radius: 5px; font-size: 12px;">
        <b>Confidence Intervals</b><br>
        <span style="color: blue; opacity: 0.4;">●</span> 50% CI (innermost)<br>
        <span style="color: blue; opacity: 0.3;">●</span> 80% CI<br>
        <span style="color: blue; opacity: 0.2;">●</span> 95% CI (outermost)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


def geocode_address(address: str, model_type: str) -> tuple[str, folium.Map]:
    """Geocode an NYC address and return coordinates with map.

    :param address: NYC address string.
    :param model_type: Model type: 'Regression' or 'MDN (Probabilistic)'.
    :return: Tuple of (coordinates text, Folium map).
    """
    if not address or not address.strip():
        empty_map = folium.Map(location=NYC_CENTER, zoom_start=11, tiles="CartoDB positron")
        return "Please enter an NYC address", empty_map

    # Determine model type from dropdown
    model_key = "mdn" if "MDN" in model_type else "regression"

    # Get inferencer (lazy loading for MDN)
    inferencer = get_inferencer(model_key)

    if model_key == "mdn":
        # MDN prediction with confidence intervals
        results = inferencer.predict_mdn_raw(address.strip())
        lat, lon = float(results["lat"][0]), float(results["lon"][0])
        sigma_lat, sigma_lon = float(results["sigma_lat"][0]), float(results["sigma_lon"][0])

        # Convert sigma to meters
        sigma_lat_m = sigma_lat * DEG_LAT_TO_M
        sigma_lon_m = sigma_lon * DEG_LON_TO_M
        sigma_max_m = max(sigma_lat_m, sigma_lon_m)

        # Format output with uncertainty info
        coords_text = f"""### Predicted Coordinates (MDN)

**Latitude:** {lat:.6f}
**Longitude:** {lon:.6f}

**Uncertainty (σ):**
- σ_lat: {sigma_lat:.6f}° ({sigma_lat_m:.0f}m)
- σ_lon: {sigma_lon:.6f}° ({sigma_lon_m:.0f}m)

**Confidence Intervals:**
- 50% CI: ±{0.6745 * sigma_max_m:.0f}m
- 80% CI: ±{1.2816 * sigma_max_m:.0f}m
- 95% CI: ±{1.9600 * sigma_max_m:.0f}m

[Open in Google Maps](https://www.google.com/maps?q={lat},{lon})
"""
        pred_map = create_mdn_map(lat, lon, sigma_lat, sigma_lon, address)
    else:
        # Regression prediction
        lat, lon = inferencer.predict(address.strip())
        lat, lon = float(lat[0]), float(lon[0])

        coords_text = f"""### Predicted Coordinates

**Latitude:** {lat:.6f}
**Longitude:** {lon:.6f}

[Open in Google Maps](https://www.google.com/maps?q={lat},{lon})
"""
        pred_map = create_regression_map(lat, lon, address)

    return coords_text, pred_map


# Example addresses for the interface
EXAMPLE_ADDRESSES = [
    ["350 5th Avenue, Manhattan, NY 10118", "Regression"],
    ["1 World Trade Center, Manhattan, NY 10007", "Regression"],
    ["200 Eastern Parkway, Brooklyn, NY 11238", "MDN (Probabilistic)"],
    ["123-01 Roosevelt Avenue, Queens, NY 11368", "MDN (Probabilistic)"],
    ["1000 Richmond Terrace, Staten Island, NY 10301", "Regression"],
    ["161st Street, Bronx, NY 10451", "MDN (Probabilistic)"],
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

**Models:**
- **Regression:** Direct lat/lon prediction (faster)
- **MDN (Probabilistic):** Mixture Density Network with uncertainty estimates and confidence intervals
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            address_input = gr.Textbox(
                label="NYC Address",
                placeholder="e.g., 350 5th Avenue, Manhattan, NY 10118",
                lines=2,
            )
            model_selector = gr.Dropdown(
                label="Model Type",
                choices=["Regression", "MDN (Probabilistic)"],
                value="Regression",
            )
            geocode_btn = gr.Button("Geocode", variant="primary")

            gr.Examples(
                examples=EXAMPLE_ADDRESSES,
                inputs=[address_input, model_selector],
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
        inputs=[address_input, model_selector],
        outputs=[coords_output, map_output],
    )

    address_input.submit(
        fn=geocode_address,
        inputs=[address_input, model_selector],
        outputs=[coords_output, map_output],
    )

    gr.Markdown(
        """
---
**About:** This app uses tiny BERT models (`google/bert_uncased_L-2_H-128_A-2`) fine-tuned on NYC address data.

**Model Types:**
- **Regression:** Standard model that directly predicts latitude/longitude coordinates
- **MDN:** Mixture Density Network that outputs a probability distribution, enabling uncertainty quantification and confidence intervals

**Limitations:**
- Only trained on NYC addresses - may not work for other locations
- Accuracy varies by borough and address format
- Best results with full addresses including borough and ZIP code
        """
    )


if __name__ == "__main__":
    demo.launch()
