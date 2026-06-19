The Concept:
Singapore's Urban Heat Island (UHI) effect is a critical policy focus. The Global Streetscapes dataset contains over 300 attributes, including specific metadata on greenery, lighting conditions, and view directions at the pedestrian level. By extracting sky-view factors and tree-canopy coverage from this imagery, you can build a high-resolution "Shade Index" for the entire island.

The Value Proposition:
Top-down satellite imagery often misses the true pedestrian experience (e.g., covered linkways, specific tree angles). Street-level vision models allow planners to pinpoint exact corridors lacking shade to prioritize NParks' tree-planting initiatives. Furthermore, this index could be integrated into the OneMap application to offer citizens "Cool Routes"—walking paths optimized for maximum shade during the afternoon heat.


Architecture:
* Frontend: HTML and CSS
* Backend: FastAPI
* Button clicked -> run score.py on images in data/images (make sure to guard for no images) and output in scores.csv
* Inference shape, should run in a batch and feel free to suggest alternative prompts:
```python
import os
from google import genai
from PIL import Image

# 1. Initialize the client (Make sure GEMINI_API_KEY is in your environment variables)
client = genai.Client()

# 2. Load your local image
image = Image.open("data/images/<name>.jpeg")

# 3. Ping the model (Using the fast, free Flash model)
response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents=[
        image, 
        '''Turn this image into JSON, example: {
  "pedestrian_shade_score": 0.72,
  "shade_sources": ["street_trees", "building_overhang"],
  "confidence": "high",
  "reasoning": "Dense canopy over left sidewalk; building shadow covers right side."
}'''
    ]
)

# 4. Print the result
print(response.text)
```
* Join with existing filtered streetscapes.csv to make geodataframe
* Display folium map, shape:
```python
import folium
sg_map = folium.Map(location=[1.3521, 103.8198], zoom_start=12, tiles='OpenStreetMap')

for index, row in gdf.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']]
        # colour based on pedestrian_shade_score
        # clicking on each point shows the score, sources, confidence and reasoning. If possible, include the image as well
    ).add_to(sg_map)
```