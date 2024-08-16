import base64
import io

import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image

from kiltergpt.utils import Plotter

# Sample Data (replace with your DataFrame)
df = pd.read_csv("data/generated_climbs.csv")
plotter = Plotter()
# Sample Image Frames (replace with your image loading logic)

app = dash.Dash(__name__)

app.layout = html.Div(
    [
        html.Div(  # Main container for plot and image+text
            [
                html.Div(dcc.Graph(id="scatter-plot"), style={"width": "50%"}),
                html.Div(
                    [
                        html.Img(id="image-display"),
                        html.Div(id="image-text", style={"font-size": "30px"}),
                    ],  # Container for the text
                    style={"width": "50%"},
                ),
            ],
            style={"display": "flex"},
        )
    ]
)


@app.callback(
    Output("image-display", "src"),
    Output("image-text", "children"),  # Output for text under image
    Input("scatter-plot", "clickData"),
)
def display_image(clickData):
    if clickData:
        point_index = clickData["points"][0]["pointIndex"]
        row = df.iloc[point_index]
        img = plotter.plot_climb(row["frames"])
        img = Image.fromarray((img).astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        text = f"{row['name']}, {row['font_grade']} @ {row['angle']}"
        return f"data:image/png;base64,{encoded_image}", text

    return "", ""


@app.callback(Output("scatter-plot", "figure"), Input("scatter-plot", "clickData"))  # Trigger on click
def update_scatter(clickData):
    return {
        "data": [
            go.Scatter(
                x=df["x_gpt"],
                y=df["y_gpt"],
                marker_color=df["difficulty_average"],
                marker=dict(colorscale="Viridis", opacity=0.9, size=4),
                # marker_symbol="x",
                mode="markers",
                hovertemplate="<b>Prompt: %{customdata[0]}</b><br>Grade: %{customdata[1]}<br>Angle: %{customdata[2]}",  # Add this line
                customdata=df[["prompt", "angle", "grade"]].values,  # Add this line
            )
        ],
        "layout": go.Layout(
            hovermode="closest",  # Ensure only closest point triggers hover
            title="UMAP of all 40 degree climbs",
            width=1200,
            height=800,
            uirevision=True,  # Add this line
        ),
    }


if __name__ == "__main__":
    app.run_server(debug=True)
