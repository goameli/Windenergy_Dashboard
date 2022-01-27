import os
#os.chdir(r"C:/Users/DEMEGUE/Documents/04_Deployment_Flask/Layout-Dashboard")

import dash
from datetime import datetime as dt
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

import plotly.graph_objects as go
import plotly.express as px

from functions import *
from windrose import WindroseAxes
from predictive import *

######Predictive data load#####
###############################

rel_path_to_turbine_data = "./data/Turbine_Data.csv"
data_stream = TurbineDataStream(rel_path_to_turbine_data)
stream_array = []
for data in data_stream:
    data['Unnamed: 0']
    data['ActivePower_t0']
    data['ActivePower_t1']
    data['ActivePower_t2']
    data['ActivePower_t3']
    data['ActivePower_t4']
    data['ActivePower_t5']
    data['ActivePower_t6']
    data['avg_WindSpeed']
    stream_array.append([data['Unnamed: 0'], data['ActivePower_t0'], data['ActivePower_t1'],  data['ActivePower_t2'],  data['ActivePower_t3'], data['ActivePower_t4'], data['ActivePower_t5'],  data['ActivePower_t6'],  data['avg_WindSpeed']])

columns = ['Unnamed: 0', 'ActivePower_t0', 'ActivePower_t1', 'ActivePower_t2', 'ActivePower_t3', 'ActivePower_t4', 'ActivePower_t5', 'ActivePower_t6', 'avg_WindSpeed']
df_predicted = pd.DataFrame(stream_array, columns=columns)
predicted_df = make_datetime_from_columns(df_predicted, input_column='Unnamed: 0')
cutted_predicted_df = cut_timeframe_from_df(predicted_df, "2020-03-01", "2020-03-31", "00:00:00", "00:00:00")[0]


######Process data load#####
############################

df = get_data()
df = make_datetime_from_columns(df, input_column="Unnamed: 0")
df_wind_rose = px.data.wind()
print(df)

df_effective = get_data()
df_effective = make_datetime_from_columns(df_effective, input_column="Unnamed: 0")
df_effective.reset_index(inplace=True)
#df_effective["year"] = df_effective['timestamp'].dt.year
df_effective["month"] = df_effective['timestamp'].dt.month
#df_effective['day'] = df_effective['timestamp'].dt.day
#df_effective['hour'] = df_effective['timestamp'].dt.hour
df_monthly = df_effective.groupby(df_effective['month']).mean()
df_monthly.reset_index(inplace=True)
print(df_monthly)


######Create app#####
#####################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])


#####Frontend Elements#####
###########################

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
        calendar_orientation='horizontal',
        day_size=39,
        first_day_of_week=1,
        start_date_placeholder_text="Start Date",
        end_date_placeholder_text="End Date",
        min_date_allowed=dt(2018, 1, 1),
        max_date_allowed=dt(2021, 9, 19),
        start_date=dt(2020, 3, 1).date(),
        end_date=dt(2020, 3, 30).date(),

        persistence=True,
        persisted_props=['start_date'],
        persistence_type='local',
        updatemode="bothdates",
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


#####FIGURES#####
#################

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


fig_windrose = fig = px.bar_polar(df_wind_rose, r="frequency", theta="direction",
                   color="strength", template="plotly_dark",
                   color_discrete_sequence= px.colors.sequential.Plasma_r)

ax = WindroseAxes.from_ax()
fig_windrose_classic = ax.bar(df["WindDirection"], df['WindSpeed'], normed=True, opening=0.8, edgecolor='white')

drop_down_correlation1 = dcc.Dropdown(
        id='feature_list_correlation1',
        placeholder="Select a Feature",
        options=[{'label': col, "value": col}
            for col in df.columns],
        value='ActivePower',
        style={"width":"55%"}
    )


drop_down_correlation2 = dcc.Dropdown(
        id='feature_list_correlation2',
        placeholder="Select a Feature",
        options=[{'label': col, "value": col}
            for col in df.columns],
        value='WindSpeed',
        style={"width":"55%"}
    )


fig_correlation = go.Figure([go.Scatter(x=df["ActivePower"], y=df["WindSpeed"], mode='lines+markers', marker = {'color' : '#d90000'})])

#fig_monthly = px.bar(df_monthly, x="month", y="ActivePower", barmode="group")

fig_monthly = go.Figure([
    go.Bar(
        x=df_monthly["month"], 
        y=df_monthly["ActivePower"])
])


"""predictive_fig = go.Figure()

predictive_fig.update_layout(
    title_text="Overview of Real/Predicted Power"
)

# Add traces
predictive_fig.add_trace(go.Scatter(x=selected_df.index, y=selected_df['ActivePower'],
                    mode='markers',
                    name='Real'))
predictive_fig.add_trace(go.Scatter(x=selected_df.index, y=cutted_predicted_df["ActivePower_t6"],
                    mode='lines+markers',
                    name='Predicted'))

predictive_fig.update_layout(
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
)"""


######LAYOUT#####
#################

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2(html.B("Dashboard-Prototyp for Wind Turbine Performance Analysis"),
                            className='text-center mt-4 mb-5',
                            style={'color': 'Gray', 'text-decoration': 'None', 'text-align': 'center'}
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
        ),
        dbc.Col(html.H4(html.B("KPI Visualization"), className="text-left"), style={'color':'gray'})
    ]),
    dbc.Row([
        dcc.Graph(
            id="gauge_min",
            style={"width":"50%"}
        ),
        dcc.Graph(
            id="gauge_max",
            style={"width":"0%"}
        ),
    ]),
    dbc.Row([
        dcc.Graph(
            id="gauge_average",
            style={"width":"50%"}
        ),
        dcc.Graph(
            id="gauge_median",
            style={"width":"0%"}
        ),
    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Wind Direction Analysis with WindRose Visualization"), className="text-left"), style={'color':'gray'}),
        dcc.Graph(
            id="wind_rose",
            figure=fig_windrose
        )

    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Feature Selection for Correlation Analysis"), className="text-left"), style={'color':'gray'}),
    ]),
    dbc.Row([
        drop_down_correlation1,
        drop_down_correlation2,
    ]),
    dbc.Row([
        dcc.Graph(
            id="scatter_correlation",
        )

    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Forecasting"), className="text-left"), style={'color':'gray'}),
    ]),
    dbc.Row([
        dcc.Graph(
            id="predictive_fig",
        )
    ]),
    dbc.Row([
        dbc.Col(html.H4(html.B("Monthly Power Overview"), className="text-left"), style={'color':'gray'}),
        dcc.Graph(
            id="monthly_figure",
            figure={
                "data": [
                    go.Bar(
                        x=df_monthly["month"],
                        y=df_monthly["ActivePower"],
                    )
                ]
            }
        )
    ]),
    dbc.Row([
        card,
        card,
    ]),


], fluid=True)


#####Callbacks#####
###################

@app.callback(
    dash.dependencies.Output('plotly_scatter_graph', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list', 'value')]
)
def update_output(start_date, end_date, selected_feature):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig = go.Figure([go.Scatter(x=dff.index, y=dff[selected_feature])])

    updated_fig.update_layout(
    title_text="Overview for {}".format(selected_feature),
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

@app.callback(
    dash.dependencies.Output('gauge_average', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list', 'value')])
def update_gauge(start_date, end_date, selected_feature):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig_average =  go.Figure(go.Indicator(
        mode = "gauge+number",
        value = np.mean(dff[selected_feature]),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average {}".format(selected_feature)}))

    return updated_fig_average


@app.callback(
    dash.dependencies.Output('gauge_min', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list', 'value')])
def update_gauge(start_date, end_date, selected_feature):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig_average =  go.Figure(go.Indicator(
        mode = "gauge+number",
        value = np.min(dff[selected_feature]),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Min {}".format(selected_feature)}))

    return updated_fig_average


@app.callback(
    dash.dependencies.Output('gauge_max', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list', 'value')])
def update_gauge(start_date, end_date, selected_feature):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig_average =  go.Figure(go.Indicator(
        mode = "gauge+number",
        value = np.max(dff[selected_feature]),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Max {}".format(selected_feature)}))

    return updated_fig_average


@app.callback(
    dash.dependencies.Output('gauge_median', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list', 'value')])
def update_gauge(start_date, end_date, selected_feature):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig_average =  go.Figure(go.Indicator(
        mode = "gauge+number",
        value = np.median(dff[selected_feature]),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Median {}".format(selected_feature)}))

    return updated_fig_average


'''@app.callback(
    dash.dependencies.Output('scatter_correlation', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list', 'value')])
def update_gauge(start_date, end_date, selected_feature):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig_correlation = go.Figure([go.Scatter(x=dff["WindSpeed"], y=dff["ActivePower"], mode='lines+markers', marker = {'color' : '#d90000'})])
    
    updated_fig_correlation.update_layout(
    title_text="[ActivePower - WindSpeed] ",
    xaxis_title = "WindSpeed[m/s]",
    yaxis_title = "ActivePower[kW]"
    )
    return updated_fig_correlation'''


@app.callback(
    dash.dependencies.Output('scatter_correlation', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date'),
    dash.dependencies.Input('feature_list_correlation1', 'value'),
    dash.dependencies.Input('feature_list_correlation2', 'value')])
def update_gauge(start_date, end_date, selected_feature1, selected_feature2):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    updated_fig_correlation = go.Figure([go.Scatter(x=dff[selected_feature1], y=dff[selected_feature2], mode='lines+markers', marker = {'color' : '#d90000'})])
    
    updated_fig_correlation.update_layout(
    title_text="[ActivePower - WindSpeed]",
    xaxis_title = selected_feature1,
    yaxis_title = selected_feature2
    )

    if selected_feature1 == "WindSpeed" and selected_feature2 == "ActivePower": 
        updated_fig_correlation.update_layout(shapes=[
        dict(
        type= 'line',
        yref= 'paper', y0= 0, y1= 1,
        xref= 'x', x0= 3.5, x1= 3.5
        )
        ])

    return updated_fig_correlation


@app.callback(
    dash.dependencies.Output('predictive_fig', 'figure'),
    [dash.dependencies.Input('date-picker-range', 'start_date'),
    dash.dependencies.Input('date-picker-range', 'end_date')])
def predictive_update(start_date, end_date):
    analysis_start_date = start_date + " " + "00:00:00" + "+00:00"
    analysis_end_date = end_date + " " + "00:00:00" + "+00:00"
    dff = df[analysis_start_date:analysis_end_date]

    predictive_fig = go.Figure()

    predictive_fig.update_layout(
        title_text="Overview of Real/Predicted Power"
    )

    # Add traces
    predictive_fig.add_trace(go.Scatter(x=dff.index, y=dff['ActivePower'],
                        mode='markers',
                        name='Real'))
    predictive_fig.add_trace(go.Scatter(x=dff.index, y=cutted_predicted_df["ActivePower_t6"],
                        mode='lines+markers',
                        name='Predicted'))

    predictive_fig.update_layout(
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
    return predictive_fig


'''@app.callback(
    dash.dependencies.Output('plotly_scatter_graph', 'figure'),
    [dash.dependencies.Input('feature_list', 'value')]
)
def change_visualization_reg_feature(selected_feature):
    #df = pd.read_csv("data/Turbine_Data.csv")
    updated_fig = go.Figure([go.Scatter(x=df.index, y=df[selected_feature])])

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

    return updated_fig'''


if __name__ == '__main__':
    app.run_server(debug=True)