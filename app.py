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

average_confirmed_days = df_quar[df_quar['Confirmed'] == 'Confirmed']['Duration'].mean()
max_confirmed_days = df_quar[df_quar['Confirmed'] == 'Confirmed']['Duration'].max()

df_quar_wuh = df_quar[df_quar['Place'] == 'Wuhan']

line_height = 30
# print(df_quar['Task'].unique())
#
# df_quar_lim = df_quar[df_quar['Country'].isin(['China', 'United Kingdom', 'France', 'Italy', 'Spain', 'United States', 'Iran', ])]
#
# df_gantt_ls = []
#
# colors = {}
# for i in range(256):
#     colors[str(i) + 'r'] = matplotlib.cm.get_cmap('Reds')(i / 255)
#     colors[str(i) + 'b'] = matplotlib.cm.get_cmap('Greens')(i / 255)
#
# for col_name in ['China - Hubei', 'United Kingdom - ', 'France - ', 'Italy - ', 'Iran - ', 'Spain - ']:
#     print(col_name)
#     for n in range(int((df_quar['Finish'].max() - df_deaths_pd.index.min()).days)):
#         date_select = df_deaths_pd.index.min() + pd.offsets.Day(n)
#         in_quarantine = False
#         if col_name in df_quar['Task'].unique():
#             df_quar_select = df_quar[df_quar['Task'] == col_name].iloc[0]
#             if (date_select >= df_quar_select['Start']) and (date_select <= df_quar_select['Finish']):
#                 in_quarantine = True
#         if date_select in df_deaths_pd.index:
#             val = df_deaths_pd.loc[date_select, col_name]
#             print(val, np.isnan(val))
#             if not np.isnan(val):
#                 if in_quarantine:
#                     col = str(int((val/3/793*255).round()+10)) + 'b'
#                 else:
#                     col = str(int((val / 3 /793 * 255).round()+10)) + 'r'
#                 df_gantt_ls.append(
#                     {'Task': col_name, 'Start': date_select, 'Finish': date_select + pd.offsets.Day(1), 'Deaths': col})
#         else:
#             if in_quarantine:
#                 col = '50b'
#                 df_gantt_ls.append(
#                         {'Task': col_name, 'Start': date_select, 'Finish': date_select + pd.offsets.Day(1), 'Deaths': col})
#
# df_gantt = pd.DataFrame(df_gantt_ls)
# # df_gantt['Deaths'] = (df_gantt['Deaths'].fillna(0) / 242 * 255).round()
# # print(df_gantt)
# gantt = ff.create_gantt(df_gantt, index_col='Deaths', colors=colors, group_tasks=True, show_colorbar=True)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(children=[
dcc.Graph(
        figure={
            'data': [
                        go.Scatter(
                            x=[pd.datetime.now().date(), pd.datetime.now().date()],
                            y=[0, len(df_quar) * line_height],
                            mode='lines',
                            name='Today',
                            line={
                                'width': 2,
                                'color': 'rgb(0,0,0)'
                            }
                        )
                    ] + [
                        go.Scatter(
                            x=[row['Start'], row['PEnd']],
                            y=[index * line_height, index * line_height],
                            mode='lines',
                            name=row['Name'],
                            line={
                                'width': 2,
                                'color': 'rgb(128,128,128)',
                                'dash': 'dash'
                            }
                        ) for index, row in df_quar.iterrows()
                    ] + [
                        go.Scatter(
                            x=[row['Start'], row['Finish']],
                            y=[index * line_height, index * line_height],
                            mode='lines',
                            name=row['Name'],
                            line={
                                'width': 4 if row['Level'] == 'National' else 2,
                                'color': 'rgb(0,255,0)' if row['Status'] == 'Ended' else 'rgb(255,0,0)'
                            }
                        ) for index, row in df_quar.dropna(subset=['Finish']).iterrows()
                    ],
            'layout': {
                'height': 1200,
                'margin': {'l': 40, 'b': 30, 't': 0, 'r': 10},
                'annotations': [],
                'xaxis': {
                    'showline': False,
                    'zeroline': False,
                    'tickformat': '%B \'%y',
                    # 'tickvals': [df_quar['Start'].min() + pd.offsets.MonthBegin(i-1) for i in range(7)],
                },
                'yaxis': {
                    'showgrid': False,
                    'zeroline': False,
                    'showticklabels': False
                },
                'showlegend': False
            }
        }
    ),
    dcc.Graph(
        figure={
            'data': [
                        go.Scatter(
                            x=[pd.datetime.now().date(), pd.datetime.now().date()],
                            y=[0, len(df_quar) * line_height],
                            mode='lines',
                            name='Today',
                            line={
                                'width': 2,
                                'color': 'rgb(0,0,0)'
                            }
                        )
                    ] + [
                        go.Scatter(
                            x=[row['Start'], row['PEnd']],
                            y=[index * line_height, index * line_height],
                            mode='lines',
                            name=row['Name'],
                            line={
                                'width': 2,
                                'color': 'rgb(128,128,128)',
                                'dash': 'dash'
                            }
                        ) for index, row in df_quar.iterrows()
                    ] + [
                        go.Scatter(
                            x=[row['Start'], row['Finish']],
                            y=[index * line_height, index * line_height],
                            mode='lines',
                            name=row['Name'],
                            line={
                                'width': 4 if row['Level'] == 'National' else 2,
                                'color': 'rgb(0,255,0)' if row['Status'] == 'Ended' else 'rgb(255,0,0)'
                            }
                        ) for index, row in df_quar.dropna(subset=['Finish']).iterrows()
                    ],
            'layout': {
                'height': 1200,
                'margin': {'l': 40, 'b': 30, 't': 0, 'r': 10},
                'annotations': [{
                    'x': row['Finish'], 'y': index * line_height, 'xref': 'x', 'yref': 'y', 'showarrow': False,
                    'font': {'size': 12},
                    'xanchor': 'left',
                    'text': '⚠️'
                } for index, row in df_quar.dropna(subset=['Finish']).iterrows()] + [{
                    'x': row['Start'], 'y': index * line_height, 'xref': 'x', 'yref': 'y', 'showarrow': False,
                    'font': {'size': 12},
                    'xanchor': 'right',
                    'yanchor': 'middle',

                    'text': row['CCE'] + ' ' + row['Name']
                } for index, row in df_quar.iterrows()],
                'xaxis': {
                    'showline': False,
                    'zeroline': False,
                    'tickformat': '%B \'%y',
                    'tickvals': [df_quar['Start'].min() + pd.offsets.MonthBegin(i-1) for i in range(7)],
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
