# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from evolvemb import list_new_tokens, list_multiple_meanings_tokens, list_semantic_shift_tokens, plot_emb_over_time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'EvolvEmb'
server = app.server

colors = {
    'background': '#ffffff',
    'text': '#111111'
}

# load embeddings
snapshot_emb = pickle.load(open("snapshot_emb.pkl", "rb"))
snapshots = sorted(snapshot_emb)


# get the most changed tokens
def get_most_changed_tokens(k=25):
    # check which tokens are new, i.e., started with a zero embedding
    new_tokens = list_new_tokens(snapshot_emb)
    new_tokens = ", ".join([f"{t[0]} ({t[1]})" for t in new_tokens[:k]])
    # check which tokens have a general meaning change somewhere, i.e., mostly words with multiple meanings
    multiple_meanings_tokens = list_multiple_meanings_tokens(snapshot_emb)
    multiple_meanings_tokens = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in multiple_meanings_tokens[:k]])
    # check which tokens underwent an actual semantic shift (i.e., continuous change, no seasonal patterns)
    semantic_shift_tokens = list_semantic_shift_tokens(snapshot_emb)
    example_token = semantic_shift_tokens[0][0]
    semantic_shift_tokens = ", ".join([f"{t[0]} ({t[1]:.3f})" for t in semantic_shift_tokens[:k]])
    return example_token, new_tokens, multiple_meanings_tokens, semantic_shift_tokens


example_token, new_tokens, multiple_meanings_tokens, semantic_shift_tokens = get_most_changed_tokens(k=25)


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

    html.Div([html.B("New words coined during the time period (and their counts): "), new_tokens],
             style={
                'textAlign': 'left',
                'color': colors['text']
             }),
    html.Br(),

    html.Div([html.B("Words with multiple meanings in general, possibly exhibiting seasonal trends (and the minimum cosine similarity score between their embedding snapshots): "), multiple_meanings_tokens],
             style={
                'textAlign': 'left',
                'color': colors['text']
             }),
    html.Br(),

    html.Div([html.B("Words with with a genuine continuous semantic shift (and our semantic shift score): "), semantic_shift_tokens],
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
    ])
])


# figures are loaded based on the text input
@app.callback(
    Output('input-validation', 'children'),
    Output('timeline-graph', 'figure'),
    Output('pca-graph', 'figure'),
    Input('input-token', 'value'))
def generate_figs(token):
    fig_time, fig_pca = plot_emb_over_time(snapshot_emb, token)
    # no update if token wasn't found
    if fig_time is None:
        return f"Warning: token '{token}' unknown", dash.no_update, dash.no_update
    return f"Results for: '{token}'", fig_time, fig_pca


if __name__ == '__main__':
    app.run_server(debug=True)
