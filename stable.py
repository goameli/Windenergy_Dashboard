import dash
from datetime import date
import pandas as pd
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

import plotly.graph_objects as go

# load data
df = pd.read_csv('data/Turbine_Data.csv', parse_dates=True) #index_col=0
#df.index = pd.to_datetime(df['Date'])


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])

card = dbc.Card(
                dbc.CardBody(
                    [
                        html.H4(html.B("card title"), className="card-title"),
                        html.Hr(),
                        html.H2(html.B("card text"),className='card-text',style={'color':'green'}),
                    ]
                ),style={"width": "18rem","border-radius":"2%","background":"PowderBlue"}
            )

date_seletion_card = dcc.DatePickerRange(
        id='date-picker-range',
        start_date_placeholder_text="Start Period",
        end_date_placeholder_text="End Period",
        min_date_allowed=date(2020, 1, 1),
        max_date_allowed=date(2022, 9, 19),
        #initial_visible_month=date(2017, 8, 5),
        #end_date=date(2017, 8, 25),
        style={"width":"30%"}
    )

drop_down_feature = dcc.Dropdown(
        id='feature_list',
        placeholder="Select a Feature",
        options=[{'label': col, "value": col}
            for col in df.columns],
        value='ActivePower',
        style={"width":"55%"}
    )

input_box = dcc.Input(
    id="time-input",
    type="text",
    placeholder="00:00:00",
    value="00:00:00",
    style={"width":"20%"}
)


# FIGURES

fig_scatter = go.Figure([
    go.Scatter(
        x=df.index,
        y=df["ActivePower"],
        mode="lines+markers",
        #marker={'color':'#d90000'},
    )
])

fig_scatter.update_layout(
    title_text="Overview for Data-Time Range",
)

fig_scatter.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# LAYOUT
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2(html.B("Dashboard-Prototyp for Wind Turbine Performance Analysis"),
                            className='text-center mt-4 mb-5',
                            style={'color': 'Black', 'text-decoration': 'None', 'text-align': 'center'}
                        )
        )
    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Select Date"), className="text-left"), style={'color':'gray'})
    ]),
    dbc.Row([
        date_seletion_card
    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Select Time"), className="text-left"), style={'color':'gray'})
    ]),
    dbc.Row([
        input_box
    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Select Feature for analysis"), className="text-left"), style={'color':'gray'})
    ]),
    dbc.Row([
        drop_down_feature
    ]),
    dbc.Row([
        dcc.Graph(
            id="plotly_scatter_graph",
            figure={
                "data":[
                    {
                        "x": df["Unnamed: 0"],
                        "y": df["ActivePower"],
                        "type": "lines",
                    }
                ],
                "layout": {"title":{"text":"Overview for Data-Time Range"}},
                'layout': go.Layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )

                )
            }
        )
    ]),
    dbc.Row([
        card,
        card,
    ]),


],fluid=True)


# Callbacks

@app.callback(
    dash.dependencies.Output('plotly_scatter_graph', 'figure'),
    [dash.dependencies.Input('feature_list', 'value')]
)
def change_visualization_reg_feature(selected_feature):
    df = pd.read_csv("data/Turbine_Data.csv")
    updated_fig = go.Figure([go.Scatter(x=df['Unnamed: 0'], y=df[selected_feature])])

    updated_fig.update_layout(
    title_text="Overview for Data-Time Range",
    )

    updated_fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

    return updated_fig



if __name__ == '__main__':
    app.run_server(debug=True)