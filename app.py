import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.figure_factory as ff

from dash.dependencies import Input, Output, State
import pandas as pd
import pycountry
import flag

pd.set_option('display.max_columns', 500)
from datetime import datetime as dt
import numpy as np
import dash_table
import math
import json
import matplotlib

# df_confirmed_csv = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv', )
# df_deaths_csv = pd.read_csv('COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
#
# country = df_confirmed_csv.T.iloc[1]
# place = df_confirmed_csv.T.iloc[0].fillna('')
# names = country + ' - ' + place
#
# df_confirmed = df_confirmed_csv.T.iloc[5:]
# df_confirmed.index = pd.to_datetime(df_confirmed.index, format='%m/%d/%y')
# df_confirmed.columns = names
#
# df_deaths = df_deaths_csv.T.iloc[5:]
# df_deaths.index = pd.to_datetime(df_deaths.index, format='%m/%d/%y')
# df_deaths.columns = names
#
# df_deaths_pd = df_deaths.diff().replace(0, np.nan).interpolate()
#
# select_lim = (df_deaths.sum() > 1000)
# df_confirmed_lim = df_confirmed.loc[:,select_lim]
# df_deaths_lim = df_deaths.loc[:,select_lim]
# df_deaths_pd_lim = df_deaths_pd.loc[:,select_lim]
#
# print(df_confirmed_lim.head())
# print(df_confirmed_lim.columns.values)
df_quar = pd.read_csv('quarantine_dates.csv')
df_quar['Start'] = pd.to_datetime(df_quar['Start date'], format='%Y-%m-%d')
df_quar['Finish'] = pd.to_datetime(df_quar['End date'], format='%Y-%m-%d')
df_quar['Status'] = 'In Progress'
df_quar.loc[df_quar['Finish'].isna(), 'Status'] = 'TBC'
df_quar.loc[df_quar['Finish'] <= pd.datetime.now(), 'Status'] = 'Ended'
df_quar['Place'] = df_quar['Place']
df_quar['Name'] = df_quar['Country'] + (' (' + df_quar['Place'] + ')').fillna('')
df_quar['Duration'] = df_quar['Finish'] - df_quar['Start']
df_quar['PEnd'] = df_quar['Start'] + pd.offsets.Day(76)
df_quar['Country Start'] = df_quar['Start'].groupby(df_quar['Country']).transform('min')
df_quar = df_quar.sort_values('Start', ascending=False).reset_index()
df_quar['CC'] = df_quar['Country'].apply(lambda x: pycountry.countries.search_fuzzy(x)[0].alpha_2)
df_quar['CCE'] = df_quar['CC'].apply(lambda x: flag.flag(x))
print(df_quar.head())

average_confirmed_days = df_quar[df_quar['Confirmed'] == 'Confirmed']['Duration'].mean().days
max_confirmed_days = df_quar[df_quar['Confirmed'] == 'Confirmed']['Duration'].max().days
df_quar_wuh = df_quar[df_quar['Place'] == 'Wuhan'].iloc[0]
x_points = [df_quar_wuh['Start'], df_quar_wuh['Start'] + pd.offsets.Day(21), df_quar_wuh['Start'] + pd.offsets.Day(42),
            df_quar_wuh['Start'] + pd.offsets.Day(average_confirmed_days),
            df_quar_wuh['Start'] + pd.offsets.Day(max_confirmed_days)]
x_text = ['Lockdown started️', 'Lockdown review date', 'Lockdown ended',
          '️%d Days<br>Average duration' % (average_confirmed_days),
          '%d Days<br>Maximum duration' % (max_confirmed_days)]
x_color = ['rgb(255,0,90)', 'rgb(255,165,0)', 'rgb(0,255,165)', 'rgb(173,173,173)', 'rgb(0,0,0)']
print(x_points)
line_height = 30
info_height = line_height * (len(df_quar) + 4)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
    dcc.Graph(
        figure={
            'data': [
                        go.Scatter(
                            x=[x_points[0], x_points[1]],
                            y=[info_height, info_height],
                            mode='lines',
                            line={
                                'width': 5,
                                'color': x_color[0]
                            },
                            hoverinfo='none'
                        ),
                        go.Scatter(
                            x=[x_points[1], x_points[2]],
                            y=[info_height, info_height],
                            mode='lines',
                            line={
                                'width': 5,
                                'color': x_color[2]
                            },
                            hoverinfo='none'
                        ),
                        go.Scatter(
                            x=[x_points[2], x_points[4]],
                            y=[info_height, info_height],
                            mode='lines',
                            line={
                                'width': 5,
                                'color': 'rgb(173,173,173)'
                            },
                            hoverinfo='none'
                        ),
                        go.Scatter(
                            x=x_points,
                            y=[info_height] * len(x_points),
                            mode='markers',
                            marker={
                                'size': 15,
                                'color': x_color
                            },
                            hoverinfo='none'
                        )
                    ] + [
                        go.Scatter(
                            x=[row['Start'], row['Start'] + pd.offsets.Day(max_confirmed_days)],
                            y=[index * line_height, index * line_height],
                            mode='lines',
                            line={
                                'width': 2,
                                'color': x_color[3]
                            },
                            hoverinfo='none'
                        ) for index, row in df_quar[df_quar['Confirmed'] != 'Confirmed'].iterrows()
                    ] + [
                        go.Scatter(
                            x=[row['Start'] + pd.offsets.Day(average_confirmed_days),
                               row['Start'] + pd.offsets.Day(max_confirmed_days)],
                            y=[index * line_height, index * line_height],
                            mode='markers',
                            marker={
                                'size': 10,
                                'color': [x_color[3], x_color[4]]
                            },
                            text=[

                                (row['Start'] + pd.offsets.Day(average_confirmed_days)).strftime(
                                    '%B %d') + ' - Average duration from lockdown start - ' + row['CCE'] + ' ' + row[
                                    'Name'],
                                (row['Start'] + pd.offsets.Day(max_confirmed_days)).strftime(
                                    '%B %d') + ' - Maximum duration from lockdown start - ' + row['CCE'] + ' ' + row[
                                    'Name'],
                            ],
                            hoverinfo='text'
                        ) for index, row in df_quar[df_quar['Confirmed'] != 'Confirmed'].iterrows()
                    ] + [
                        go.Scatter(
                            x=[row['Start'], row['Finish']],
                            y=[index * line_height, index * line_height],
                            mode='lines',
                            line={
                                'width': 3,
                                'color': 'rgb(0,255,165)' if row['Finish'] <= pd.datetime.now() else x_color[0],
                            },
                            hoverinfo='none'
                        ) for index, row in df_quar.iterrows()
                    ] + [
                        go.Scatter(
                            x=[row['Finish']],
                            y=[index * line_height],
                            mode='markers',
                            marker={
                                'size': 10,
                                'color': x_color[2] if row['Finish'] <= pd.datetime.now() else x_color[1],
                            },
                            text=row['Finish'].strftime(
                                '%B %d') + ' - Lockdown review date - ' + row['CCE'] + ' ' + row['Name'],
                            hoverinfo='text'
                        ) for index, row in df_quar.dropna(subset=['Finish']).iterrows()
                    ] + [
                        go.Scatter(
                            x=[row['Start']],
                            y=[index * line_height],
                            mode='markers',
                            marker={
                                'size': 10,
                                'color': x_color[0],
                            },
                            text=row['Start'].strftime(
                                '%B %d') + ' - Lockdown started - ' + row['CCE'] + ' ' + row['Name'],
                            hoverinfo='text'
                        ) for index, row in df_quar.iterrows()
                    ],
            'layout': {
                'hovermode': 'closest',
                'height': 1800,
                'margin': {'l': 40, 'b': 30, 't': 0, 'r': 10},
                'annotations': [
                                   {
                                       'x': x, 'y': info_height, 'xref': 'x', 'yref': 'y', 'showarrow': True,
                                       'arrowhead': 0, 'ax': 0, 'ay': -40, 'arrowcolor': color,
                                       'font': {'size': 14},
                                       'xanchor': 'middle',
                                       'text': text
                                   } for x, text, color in zip(x_points, x_text, x_color)
                               ] + [{
                    'x': row['Start'], 'y': index * line_height, 'xref': 'x', 'yref': 'y', 'showarrow': False,
                    'font': {'size': 15},
                    'xanchor': 'right',
                    'yanchor': 'middle',

                    'text': row['Name'] + '          '
                } for index, row in df_quar.iterrows()] + [{
                    'x': row['Start'], 'y': index * line_height, 'xref': 'x', 'yref': 'y', 'showarrow': False,
                    'font': {'size': 22},
                    'xanchor': 'right',
                    'yanchor': 'middle',

                    'text': row['CCE'] + '  '
                } for index, row in df_quar.iterrows()],
                'xaxis': {
                    'showline': False,
                    'zeroline': False,
                    'tickformat': '%B \'%y',
                    'tickvals': [df_quar['Start'].min() + pd.offsets.MonthBegin(i - 1) for i in range(7)],
                },
                'yaxis': {
                    'showgrid': False,
                    'zeroline': False,
                    'showticklabels': False
                },
                'showlegend': False
            }
        }
    )
])
#
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


if __name__ == '__main__':
    app.run_server(debug=True)
