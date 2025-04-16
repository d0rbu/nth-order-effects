import plotly.graph_objects as go
import numpy as np
import pandas as pd

data = {
    "x": 3*np.random.rand(512),
    "y": np.random.rand(512), 
    "z": np.random.random(512), 
}

df = pd.DataFrame(data)

def get_label(z):
    if z<0.25:
        return "smol dot"
    if z<0.5:
        return "ok-ish dot"
    if z<0.75:
        return "a dot"
    else:
        return "biiig dot"

df["label"] = df["z"].apply(get_label)

df["z"] = (df["z"]+1)*5

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['x'],
    y=df['y'],
    mode='markers',
    marker=dict(
        size=df['z'],
        color=df['z'],
        colorscale=[
            [0, 'rgb(78, 165, 183)'],      # Light blue
            [0.5, 'rgb(206, 192, 250)'],    # Purple
            [1, 'rgb(232, 137, 171)']       # Pink
        ],
        opacity=0.9,
    ),
        customdata=df[["label"]],
    hovertemplate="Dot category: %{customdata[0]}",
    hoverlabel=dict(namelength=0),
    showlegend=False
))


fig.update_layout(
    width=1200,
    height=400,
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False,
    margin=dict(l=0, r=0, t=0, b=0),
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        range=[0, 3]
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        scaleanchor="x",
        scaleratio=1,
        range=[0, 1]
    )
)

fig.show()

fig.write_html("fragments/banner.html", 
               include_plotlyjs=False, 
               full_html=False, 
               config={
                   'displayModeBar': False,
                   'responsive': True, 
                   'scrollZoom': False,
               })