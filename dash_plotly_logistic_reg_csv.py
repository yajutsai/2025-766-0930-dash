# Implement the same functionality as in the previous example, 
# but using dash.
import pandas as pd
import dash
from dash import dcc, html, Input, Output
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
import io
from sklearn.datasets import load_iris
import numpy as np
from sklearn.linear_model import LogisticRegression 
import base64

# Load the iris dataset
iris = load_iris()  
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])
df['target'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
# Fit a logistic regression model
model = LogisticRegression(max_iter=200)
model.fit(df.drop(columns=['target']), df['target'])
model_accuracy = model.score(df.drop(columns=['target']), df['target']) * 100
# Initialize the Dash app
app = dash.Dash(__name__)

# simple sidebar + content styles
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "22%",
    "padding": "20px",
    "background-color": "#f8f9fa",
    "overflowY": "auto",
}
CONTENT_STYLE = {
    "marginLeft": "24%",
    "padding": "20px",
}

app.layout = html.Div([
    html.Div(
        [
            html.H2("Controls", style={"marginTop": 0}),
            html.Div("Select X and Y axes, Size, and Color options."),
            html.Div([html.Label("X Axis:"), dcc.Dropdown(
                id='x-axis',
                options=[{'label': col, 'value': col} for col in df.columns[:-1]],
                value='sepal length (cm)'
            )]),
            html.Div([html.Label("Y Axis:"), dcc.Dropdown(
                id='y-axis',
                options=[{'label': col, 'value': col} for col in df.columns[:-1]],
                value='sepal width (cm)'
            )]),
            html.Div([html.Label("Size:"), dcc.Dropdown(
                id='size',
                options=[{'label': col, 'value': col} for col in df.columns[:-1]],
                value='petal length (cm)'
            )]),
            html.Div([html.Label("Color:"), dcc.Dropdown(
                id='color',
                options=[{'label': col, 'value': col} for col in df.columns],
                value='target'
            )]),
            html.Br(),
            html.Div("Upload a CSV file with the same feature columns to see predictions."),
            dcc.Upload(
                id='upload-data',
                children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginTop': '10px'
                },
                multiple=False
            ),
            html.Br(),
            html.Div(id='model-accuracy', children=html.H4(f"Model accuracy: {round(model_accuracy,2)}%")),
        ],
        style=SIDEBAR_STYLE,
    ),

    # Main content
    html.Div(
        [
            html.H1("Iris Dataset Scatter Plot with Logistic Regression Predictions"),
            html.H2('Prediction for uploaded data'),
            html.Div( id='uploaded-data-table'),
            dcc.Graph(id='scatter-plot', style={"height": "80vh"}),
        ],
        style=CONTENT_STYLE,
    ),
])
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = io.BytesIO(base64.b64decode(content_string))
    try:
        df_uploaded = pd.read_csv(decoded)
        required_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
        if all(col in df_uploaded.columns for col in required_cols):
            for col in required_cols:
                df_uploaded[col] = pd.to_numeric(df_uploaded[col], errors='coerce')
            df_uploaded.dropna(subset=required_cols, inplace=True)
            if not df_uploaded.empty:
                predictions = model.predict(df_uploaded[required_cols])
                df_uploaded['predicted_species'] = predictions
                return df_uploaded
    except Exception as e:
        print(e)
    return None
@app.callback(
    Output('scatter-plot', 'figure'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input('size', 'value'),
    Input('color', 'value'),
    Input('upload-data', 'contents')
)
def update_graph(x, y, size, color, contents):
    fig = px.scatter(df, x=x, y=y, size=size, color=color,
                     labels={x: x.replace(' (cm)', '').title(),
                             y: y.replace(' (cm)', '').title(),
                             color: 'Species' if color == 'target' else color.replace('_', ' ').title()},
                     title="Iris Dataset Scatter Plot")
    if contents:
        df_uploaded = parse_contents(contents)
        if df_uploaded is not None and not df_uploaded.empty:
            # compute marker sizes (fallback to constant if chosen size column not present)
            if size in df_uploaded.columns:
                sizes_col = pd.to_numeric(df_uploaded[size], errors='coerce').fillna(10)
                marker_sizes = np.clip(sizes_col, 5, 20).tolist()
            else:
                marker_sizes = 10
            # add uploaded points as a separate trace (distinct symbol/color)
            fig.add_trace(go.Scatter(
                x=df_uploaded[x],
                y=df_uploaded[y],
                mode='markers',
                marker=dict(size=marker_sizes, color='black', symbol='x',
                            line=dict(width=1, color='DarkSlateGrey')),
                name='Uploaded (predicted)'
            ))
    fig.update_layout(transition_duration=500)
    return fig  
@app.callback(
    Output('uploaded-data-table', 'children'),
    Input('upload-data', 'contents')
)
def display_uploaded_table(contents):
    """Return a Dash DataTable showing the uploaded CSV with model predictions."""
    if not contents:
        return html.Div("No uploaded data.")
    df_uploaded = parse_contents(contents)
    if df_uploaded is None or df_uploaded.empty:
        return html.Div("Uploaded file invalid or missing required columns.")
    return dash_table.DataTable(
        data=df_uploaded.to_dict('records'),
        columns=[{"name": c, "id": c} for c in df_uploaded.columns],
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left', 'minWidth': '100px', 'whiteSpace': 'normal'}
    )
server = app.server
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port =8080)

# run it at command line with    "python dash_plotly_logistic_reg_csv.py --show --autoreload"