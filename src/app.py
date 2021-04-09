# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from evolvemb import most_changed_tokens, analyze_emb_over_time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#ffffff',
    'text': '#111111'
}

# load embeddings
snapshot_emb = pickle.load(open("snapshot_emb.pkl", "rb"))
snapshots = sorted(snapshot_emb)


# get the most changed tokens
def get_most_changed_tokens(k=25, ignore_zeros=True):
    tokens = most_changed_tokens(snapshot_emb, ignore_zeros)
    return tokens[0][0], ", ".join([f"{t[0]} ({t[1]:.3f})" for t in tokens[:k]])


example_token, most_changed_existing = get_most_changed_tokens(k=25, ignore_zeros=True)
_, most_changed_zeros = get_most_changed_tokens(k=25, ignore_zeros=False)


# define layout
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Continuously Evolving Embeddings',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.H5(children='Explore semantic shift and word usage change over time.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Br(),

    html.Div([html.B("Words that have changed a lot during the time period: "), most_changed_existing],
             style={
                'textAlign': 'left',
                'color': colors['text']
             }),
    html.Br(),

    html.Div([html.B("Words that have changed a lot during the time period (incl. new words): "), most_changed_zeros],
             style={
                'textAlign': 'left',
                'color': colors['text']
             }),
    html.Br(),

    html.Div([html.B("Word of interest: "),
              dcc.Input(id='input-token', value=example_token, type='text')],
             style={
                'textAlign': 'center',
                'color': colors['text']
             }),
    html.Br(),
    html.Div(id="input-validation",
             style={
                'textAlign': 'left',
                'color': colors['text']
             }),
    html.Br(),
    html.Div([

        dcc.Graph(
            id='timeline-graph'
        ),

        dcc.Graph(
            id='pca-graph'
        )
    ],
        style={'columnCount': 2})
])


# figures are loaded based on the text input
@app.callback(
    Output('input-validation', 'children'),
    Output('timeline-graph', 'figure'),
    Output('pca-graph', 'figure'),
    Input('input-token', 'value'))
def generate_figs(token):
    fig_time, fig_pca = analyze_emb_over_time(snapshot_emb, token)
    # no update if token wasn't found
    if fig_time is None:
        return f"Warning: token '{token}' unknown", dash.no_update, dash.no_update
    return f"Results for: '{token}'", fig_time, fig_pca


if __name__ == '__main__':
    app.run_server(debug=True)
