import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import pycountry
import flag
import re
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 500)

import requests
from bs4 import BeautifulSoup
import numpy as np

r = requests.get("https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic")
soup = BeautifulSoup(r.content, "html.parser")
inter_table = soup.find_all("span", string="Coronavirus quarantines outside China")[0].find_parent('table')
china_table = soup.find_all("span", string="Cities under quarantine in China")[0].find_parent('table')
china_table.find_all("th", text=re.compile('^Quarantine total$'))[0].find_parent('tr').decompose()
china_table.findAll('tr')[-1].decompose()

def remove_sup(tab):
    tag_list = tab.findAll('sup')
    for t in tag_list:
        t.decompose()
    tab.findAll('tr')[0].decompose()

remove_sup(inter_table)
remove_sup(china_table)

df_inter = pd.read_html(str(inter_table))[0]
df_inter = df_inter.iloc[:-1]

df_inter = df_inter.append(pd.DataFrame([
    {'Country': 'United States', 'Place': 'Alaska', 'Start date': '2020-03-28', 'End date': '2020-04-11', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Arizona', 'Start date': '2020-03-31', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'California', 'Start date': '2020-03-19', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Colorado', 'Start date': '2020-03-26', 'End date': '2020-04-11', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Connecticut', 'Start date': '2020-03-23', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Delaware', 'Start date': '2020-03-24', 'End date': '2020-05-15', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'District of Columbia', 'Start date': '2020-04-01', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Hawaii', 'Start date': '2020-03-25', 'End date': '2020-04-30', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Idaho', 'Start date': '2020-03-25', 'End date': '2020-04-15', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Illinois', 'Start date': '2020-03-21', 'End date': '2020-04-07', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Indiana', 'Start date': '2020-03-24', 'End date': '2020-04-06', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Kansas', 'Start date': '2020-03-30', 'End date': '2020-04-19', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Louisiana', 'Start date': '2020-03-23', 'End date': '2020-04-12', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Maryland', 'Start date': '2020-03-30', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Massachusetts', 'Start date': '2020-03-24', 'End date': '2020-04-07', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Michigan', 'Start date': '2020-03-24', 'End date': '2020-04-14', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Minnesota', 'Start date': '2020-03-27', 'End date': '2020-04-10', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Montana', 'Start date': '2020-03-28', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'New Hampshire', 'Start date': '2020-03-27', 'End date': '2020-05-04', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'New Jersey', 'Start date': '2020-03-21', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'New Mexico', 'Start date': '2020-03-24', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'New York', 'Start date': '2020-03-22', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'North Carolina', 'Start date': '2020-03-30', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Ohio', 'Start date': '2020-03-23', 'End date': '2020-04-06', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Oregon', 'Start date': '2020-03-23', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Rhode Island', 'Start date': '2020-03-30', 'End date': '2020-04-26', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Virginia', 'Start date': '2020-03-30', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Vermont', 'Start date': '2020-03-25', 'End date': '2020-04-15', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Washington', 'Start date': '2020-03-25', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'West Virginia', 'Start date': '2020-03-24', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Vermont', 'Start date': '2020-03-25', 'End date': '2020-04-15', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Wisconsin', 'Start date': '2020-03-25', 'End date': '2020-04-24', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Vermont', 'Start date': '2020-03-25', 'End date': '2020-04-15', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Maryland', 'Start date': '2020-03-23', 'End date': '2020-04-07', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Nevada', 'Start date': '2020-03-17', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Virginia', 'Start date': '2020-03-24', 'End date': '2020-04-23', 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Kentucky', 'Start date': '2020-03-23', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Georgia', 'Start date': '2020-03-23', 'End date': np.nan, 'Level': 'State'},
    {'Country': 'United States', 'Place': 'Oklahoma', 'Start date': '2020-03-25', 'End date': '2020-04-30', 'Level': 'State'},
    {'Country': 'Lithuania', 'Place': 'Lithuania', 'Start date': '2020-03-16', 'End date': '2020-04-13', 'Level': 'National'},
    {'Country': 'United Arab Emirates', 'Place': 'United Arab Emirates', 'Start date': '2020-03-22', 'End date': '2020-04-05', 'Level': 'National'},
    {'Country': 'Lebanon', 'Place': 'Lebanon', 'Start date': '2020-03-22', 'End date': '2020-04-12', 'Level': 'National'},
    {'Country': 'Kuwait', 'Place': 'Kuwait', 'Start date': '2020-03-14', 'End date': np.nan, 'Level': 'National'},
    {'Country': 'India', 'Place': 'India', 'Start date': '2020-03-24', 'End date': '2020-04-14', 'Level': 'National'},
    {'Country': 'Peru', 'Place': 'Peru', 'Start date': '2020-03-16', 'End date': '2020-04-12', 'Level': 'National'},
    {'Country': 'Canada', 'Place': 'Quebec', 'Start date': '2020-03-13', 'End date': np.nan, 'Level': 'Province'},
    {'Country': 'El Salvador', 'Place': 'El Salvador', 'Start date': '2020-03-22', 'End date': np.nan, 'Level': 'National'},
    {'Country': 'Czech Republic', 'Place': 'Czech Republic', 'Start date': '2020-03-22', 'End date': '2020-04-12',
     'Level': 'National'}
]), ignore_index=True)
df_inter = df_inter.drop_duplicates(['Country', 'Place'], keep='last')
df_inter['Confirmed'] = ''

df_china = pd.read_html(str(china_table), header=0)[0]
df_china = df_china.iloc[1:-1]
df_china = df_china[['Place', 'Start date', 'End date']]
df_china['Country'] = 'China'
df_china['Level'] = 'City'
df_china['Confirmed'] = 'Confirmed'

df_quar = pd.concat((df_inter, df_china))
df_quar.loc[df_quar['Place'] == df_quar['Country'], 'Place'] = np.nan

df_quar.to_csv('deploy/lockdown_dates.csv', index=False)
df_quar.to_csv('history/lockdown_dates_%s.csv' % (pd.datetime.now().strftime('%d-%m-%y')), index=False)
df_quar_old = pd.read_csv('history/lockdown_dates_%s.csv' % ((pd.datetime.now() - pd.offsets.Day(1)).strftime('%d-%m-%y')))

print(pd.concat([df_quar, df_quar_old]).drop_duplicates(keep=False).sort_values(['Country', 'Place']))

df_quar['Start'] = pd.to_datetime(df_quar['Start date'], format='%Y-%m-%d')
df_quar['Finish'] = pd.to_datetime(df_quar['End date'], format='%Y-%m-%d')

df_quar_china_us = df_quar[df_quar['Country'].isin(['United States', 'China'])]
df_quar_other = df_quar[~df_quar['Country'].isin(['United States', 'China'])]
df_quar_other = df_quar_other.groupby('Country').agg({'Start': 'min', 'Finish': 'max', 'Place': lambda x: ', '.join(x.fillna('')), 'Level': 'first'}).reset_index()
# print(df_quar_other)
df_quar = pd.concat((df_quar_china_us, df_quar_other))

df_quar.loc[df_quar['Finish'] > (pd.datetime.now() + pd.offsets.Day(60)), 'Finish'] = np.nan
df_quar['Duration'] = df_quar['Finish'] - df_quar['Start']
df_quar['Name'] = np.where(df_quar['Level'] == 'National', df_quar['Country'], df_quar['Country'] + ' (' + df_quar['Place'] + ')')
df_quar['CC'] = df_quar['Country'].apply(lambda x: pycountry.countries.search_fuzzy(x)[0].alpha_2)
df_quar['CCE'] = df_quar['CC'].apply(lambda x: flag.flag(x))
df_quar['Complete'] = df_quar['Finish'] <= pd.datetime.now()
print(df_quar)
df_quar = df_quar.sort_values(['Start', 'Country', 'Duration'], ascending=[False, False, True]).reset_index()

average_confirmed_days = df_quar[df_quar['Confirmed'] == 'Confirmed']['Duration'].mean().days
average_review_days = df_quar[df_quar['Confirmed'] != 'Confirmed']['Duration'].mean().days
max_confirmed_days = df_quar[df_quar['Confirmed'] == 'Confirmed']['Duration'].max().days
average_to_review_days = (df_quar[df_quar['Confirmed'] != 'Confirmed']['Finish'] - pd.datetime.now()).mean().days
average_into_lockdown_days = (pd.datetime.now() - df_quar[df_quar['Confirmed'] != 'Confirmed']['Start']).mean().days
total_finished_country = df_quar[(df_quar['Confirmed'] == 'Confirmed') & (df_quar['Complete'])]['Country'].nunique()
total_finished = len(df_quar[(df_quar['Confirmed'] == 'Confirmed') & (df_quar['Complete'])])
no_end = df_quar['Finish'].isna().sum()

x_points = [0, 1, 2, 3, 4]
x_text = ['<b>Lockdowns started</b><br>%d in %d Countries<br>%d Days ago️<br>on average' % ( len(df_quar),
    df_quar['Country'].nunique(), average_into_lockdown_days),
          '<b>Lockdown reviews</b><br>%d Days from now<br>on average' % (average_review_days),
          '<b>Lockdowns ended</b><br>%d in %d Countries<br>so far' % (total_finished, total_finished_country),
          '<b>Average duration</b><br>%d Days confirmed<br>so far' % (average_confirmed_days),
          '<b>Maximum duration</b><br>%d Days confirmed<br>so far' % (max_confirmed_days)]
x_color = ['rgb(255,0,90)', 'rgb(255,165,0)', 'rgb(0,255,165)', 'rgb(223,223,223)', 'rgb(180,180,180)']
line_height = -30
info_height = 0
offset_height = -100

print(x_text, no_end)

def time_fmt(row, date, text):
    dtt = (date - pd.datetime.now()).days + 1
    if dtt == 0:
        day_str = 'Today'
    else:
        day_str = ('1 Day' if dtt * dtt == 1 else str(abs(dtt)) + ' Days') + ' ' + ('ago' if dtt < 0 else 'from now')

    return '%s %s - %s<br>%s<br>%s (%s)' % (
    row['CCE'], row['Name'], row['Level'], text, day_str, date.strftime('%d %b'))


fig2 = go.Figure(
    data=[
             go.Scatter(
                 x=[x_points[0], x_points[1]],
                 y=[info_height, info_height],
                 mode='lines',
                 line={
                     'width': 5,
                     'color': x_color[0]
                 },
                 hoverinfo='none',
                 xaxis='x1',
                 yaxis='y1'
             ),
             go.Scatter(
                 x=[x_points[1], x_points[2]],
                 y=[info_height, info_height],
                 mode='lines',
                 line={
                     'width': 5,
                     'color': x_color[2]
                 },
                 hoverinfo='none',
                 xaxis='x1',
                 yaxis='y1'
             ),
             go.Scatter(
                 x=[x_points[2], x_points[4]],
                 y=[info_height, info_height],
                 mode='lines',
                 line={
                     'width': 5,
                     'color': x_color[3]
                 },
                 hoverinfo='none',
                 xaxis='x1',
                 yaxis='y1'
             ),
             go.Scatter(
                 x=x_points,
                 y=[info_height] * len(x_points),
                 mode='markers',
                 marker={
                     'size': 15,
                     'color': x_color
                 },
                 hoverinfo='none',
                 hoverlabel={
                     'bgcolor': 'white',
                     'bordercolor': x_color[0],
                     'font': {
                         'color': 'black'
                     }
                 },
                 xaxis='x1',
                 yaxis='y1'
             )
         ] + [
             go.Scatter(
                 x=[pd.datetime.now().date(), pd.datetime.now().date()],
                 y=[0, len(df_quar) * line_height + offset_height],
                 mode='lines',
                 line={
                     'width': 2,
                     'color': 'rgb(210,210,210)'
                 },
                 hoverinfo='none',
                 xaxis='x2',
                 yaxis='y2'
             )
         ] + [
             go.Scatter(
                 x=[row['Start'], row['Start'] + pd.offsets.Day(max_confirmed_days)],
                 y=[index * line_height + offset_height, index * line_height + offset_height],
                 mode='lines',
                 line={
                     'width': 3,
                     'color': x_color[3]
                 },
                 hoverinfo='none',
                 xaxis='x2',
                 yaxis='y2'
             ) for index, row in df_quar[df_quar['Confirmed'] != 'Confirmed'].iterrows()
         ] + [
             go.Scatter(
                 x=[row['Start'] + pd.offsets.Day(average_confirmed_days),
                    row['Start'] + pd.offsets.Day(max_confirmed_days)],
                 y=[index * line_height + offset_height, index * line_height + offset_height],
                 mode='markers',
                 marker={
                     'size': 10,
                     'color': [x_color[3], x_color[4]]
                 },
                 text=[
                     time_fmt(row, row['Start'] + pd.offsets.Day(average_confirmed_days),
                              'Average duration would end'),
                     time_fmt(row, row['Start'] + pd.offsets.Day(max_confirmed_days),
                              'Maximum duration would end')
                 ],
                 hoverinfo='text',
                 hoverlabel={
                     'bgcolor': 'white',
                     'bordercolor': [x_color[3], x_color[4]],
                     'font': {
                         'color': 'black'
                     }
                 },
                 xaxis='x2',
                 yaxis='y2'
             ) for index, row in df_quar[df_quar['Confirmed'] != 'Confirmed'].iterrows()
         ] + [
             go.Scatter(
                 x=[row['Start'], row['Finish']],
                 y=[index * line_height + offset_height, index * line_height + offset_height],
                 mode='lines',
                 line={
                     'width': 3,
                     'color': x_color[2] if row['Finish'] <= pd.datetime.now() and row['Confirmed'] == 'Confirmed' else x_color[0],
                 },
                 hoverinfo='none',
                 xaxis='x2',
                 yaxis='y2'
             ) for index, row in df_quar.iterrows()
         ] + [
             go.Scatter(
                 x=[row['Finish']],
                 y=[index * line_height + offset_height],
                 mode='markers',
                 marker={
                     'size': 10,
                     'color': x_color[2] if row['Confirmed'] == 'Confirmed' else x_color[1],
                 },
                 text=time_fmt(row, row['Finish'],
                               ('Lockdown ended' if row['Finish'] <= pd.datetime.now() else 'Lockdown end confirmed') if
                               row['Confirmed'] == 'Confirmed' else 'Lockdown review'),
                 hoverinfo='text',
                 hoverlabel={
                     'bgcolor': 'white',
                     'bordercolor': x_color[2] if row['Confirmed'] == 'Confirmed' else x_color[1],
                     'font': {
                         'color': 'black'
                     }
                 },
                 xaxis='x2',
                 yaxis='y2'
             ) for index, row in df_quar.dropna(subset=['Finish']).iterrows()
         ] + [
             go.Scatter(
                 x=[row['Start']],
                 y=[index * line_height + offset_height],
                 mode='markers',
                 marker={
                     'size': 10,
                     'color': x_color[0],
                 },
                 text=time_fmt(row, row['Start'], 'Lockdown started'),
                 hoverinfo='text',
                 hoverlabel={
                     'bgcolor': 'white',
                     'bordercolor': x_color[0],
                     'font': {
                         'color': 'black'
                     }
                 },
                 xaxis='x2',
                 yaxis='y2'
             ) for index, row in df_quar.iterrows()
         ],
    layout={
        'title': {
            'text': "Global Covid-19 Lockdown Tracker",
            'y': 0.98,
            'x': 0.04,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {
                'family': 'Montserrat',
                'size': 50,
                'color': 'rgb(49, 77, 160)'
            }

        },
        'hovermode': 'closest',
        'paper_bgcolor': 'rgba(255,255,255)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'height': 2400,
        'margin': {'t': 40, 'l': 0, 'r': 0, 'b': 0},
        'annotations': [
                           {
                               'x': 0.05, 'y': 0, 'xref': 'x3', 'yref': 'y3',
                               'font': {
                                   'size': 16,
                                   'family': 'Montserrat',
                                   'color': 'rgb(39, 44, 63)'
                               },
                               'xanchor': 'left',
                               'yanchor': 'top',
                               'align': 'left',
                               'text': '<b>Last updated ' + pd.datetime.now().strftime(
                                   '%d %B %Y') + '</b><br>The most comprehensive source for how past and current lockdowns are unfolding, updated daily. Please share if you find this useful.<br>Interactive version and data download <a href="https://auravision.ai/covid19-lockdown-tracker">https://auravision.ai/covid19-lockdown-tracker</a>.',
                           },
                           {
                               'x': 1.0, 'y': 0, 'xref': 'paper', 'yref': 'paper',
                               'showarrow': False,
                               'font': {
                                   'size': 10,
                                   'family': 'Montserrat',
                                   'color': 'rgb(39, 44, 63)'
                               },
                               'xanchor': 'right',
                               'yanchor': 'bottom',
                               'align': 'right',
                               'text': 'Data sources include: <a href="https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic#In_other_countries">https://en.wikipedia.org/wiki/National_responses_to_the_2019-20_coronavirus_pandemic#In_other_countries</a>,<br><a href="https://www.businessinsider.com/countries-on-lockdown-coronavirus-italy-2020-3?r=US&IR=T">https://www.businessinsider.com/countries-on-lockdown-coronavirus-italy-2020-3?r=US&IR=T</a><br><a href="https://edition.cnn.com/2020/03/23/us/coronavirus-which-states-stay-at-home-order-trnd/index.html">https://edition.cnn.com/2020/03/23/us/coronavirus-which-states-stay-at-home-order-trnd/index.html</a>.<br><b>Aura Vision is a provider of in-store retail analytics, for more information please visit <a href="https://auravision.ai">https://auravision.ai</a></b>.'
                           }
                       ] + [
                           {
                               'x': x, 'y': 0, 'xref': 'x1', 'yref': 'y1', 'showarrow': True,
                               'arrowhead': 0, 'ax': 0, 'ay': -60, 'arrowcolor': color, 'arrowwidth': 2,
                               'font': {'size': 16},
                               'xanchor': 'center',
                               'text': text
                           } for x, text, color in zip(x_points, x_text, x_color)
                       ] + [
                           {
                               'x': pd.datetime.now().date(), 'y': -40, 'xref': 'x2', 'yref': 'y2',
                               'showarrow': False,
                               'arrowwidth': 2,
                               'font': {'size': 24},
                               'xanchor': 'right',
                               'yanchor': 'middle',
                               'text': 'Latest update '
                           }
                       ] + [{
            'x': row['Start'], 'y': index * line_height + offset_height + 5, 'xref': 'x2', 'yref': 'y2',
            'showarrow': False,
            'font': {'size': 13},
            'xanchor': 'right',
            'yanchor': 'middle',

            'text': ('<b>' + row['Name'] + '</b>' if row['Level'] == 'National' else row['Name']) + ' <span style="font-size: 18px">' + row['CCE'] + '</span>   ' + row['Start'].strftime('%d %b') + '  '
        } for index, row in df_quar.iterrows()],
        'xaxis': {
            'visible': False,
            'range': [-1, 5],
            'fixedrange': True
        },
        'yaxis': {
            'visible': False,
            'domain': [0.85, 0.95],
            'range': [-40, 80],
            'fixedrange': True
        },
        'xaxis2': {
            'position': 0.85,
            'showline': False,
            'zeroline': False,
            'tickformat': '%d %b',
            'tickvals': [pd.datetime.now().date() + pd.offsets.Day((pd.datetime.now() - (pd.datetime.now() - pd.offsets.MonthBegin(1))).days * (i - 50)) for i in range(100)],
            'tickfont': {
                'size': 16
            },
            'side': 'top',
            'gridcolor': 'rgb(230,230,230)',
            'fixedrange': True
        },
        'yaxis2': {
            'domain': [0, 0.85],
            'showgrid': False,
            'zeroline': True,
            'showticklabels': False,
            'range': [len(df_quar) * line_height + offset_height, 0],
            'zerolinewidth': 1,
            'zerolinecolor': 'rgb(128,128,128)',
            'fixedrange': True
        },
        'xaxis3': {
            'visible': False,
            'range': [0, 1],
            'fixedrange': True
        },
        'yaxis3': {
            'visible': False,
            'domain': [0.95, 1.0],
            'range': [-10, 50],
            'fixedrange': True
        },
        'showlegend': False
    }
)

fig2.write_html('deploy/index.html', auto_open=False)
fig2.write_image("deploy/image.png", width=1200, scale=1.5)

print('done')
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
#
# app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#
# app.layout = html.Div(children=[
#     html.Div(
#         className='back',
#         children=[
#             html.Img(src=app.get_asset_url('aura_vision_logo.png'), className='btyULU')
#         ]),
#     html.Div(
#         className='body',
#         children=[
#             html.H1('Global COVID-19 Lockdown Tracker'),
#             html.P(
#                 'The coronavirus outbreak is a global human tragedy, affecting hundreds of thousands of people. It is also having a growing impact on the global economy, not least retailers who are suffering an unprecendented loss in business.'),
#             html.P(
#                 [
#                     'Aura Vision\'s Global Lockdown Tracker provides business leaders with a view of quarantine timelines and typical durations as the situation evolves. This tool visualises how past and current lockdowns are unfolding using publically available information from ',
#                     html.A('Wikipedia',
#                            href="https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic#In_other_countries."),
#                     ', but cannot be used to predicted when each lockdown will end. We hope you find this tool useful, for more information please contact ',
#                     html.A('hello@auravision.ai',
#                            href="mailto:hello@auravision.ai."), '.']),
#             html.H1('As of %s:' % (pd.datetime.now().strftime('%d %B %Y'))),
#             html.Table([
#                 html.Tr([
#                     html.Td([
#                         html.H1(df_quar['Country'].nunique()),
#                         html.P('Countries in lockdown')
#                     ]),
#                     html.Td([
#                         html.H1('%d Days' % average_confirmed_days),
#                         html.P('Average lockdown duration so far')
#                     ]), html.Td([
#                         html.H1('%d Days' % max_confirmed_days),
#                         html.P('Maximum lockdown duration so far in %s' % (
#                             df_quar.loc[df_quar['Duration'].idxmax, 'Name']))
#                     ]), html.Td([
#                         html.H1('%d Days' % average_review_days),
#                         html.P('Average till lockdown review')
#                     ])
#                 ])
#             ])
#         ]),
#     html.Iframe(srcDoc='first_figure.html')
# ])
# #
# operators = [['ge ', '>='],
#              ['le ', '<='],
#              ['lt ', '<'],
#              ['gt ', '>'],
#              ['ne ', '!='],
#              ['eq ', '='],
#              ['contains '],
#              ['datestartswith ']]
#
#
# def split_filter_part(filter_part):
#     for operator_type in operators:
#         for operator in operator_type:
#             if operator in filter_part:
#                 name_part, value_part = filter_part.split(operator, 1)
#                 name = name_part[name_part.find('{') + 1: name_part.rfind('}')]
#
#                 value_part = value_part.strip()
#                 v0 = value_part[0]
#                 if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
#                     value = value_part[1: -1].replace('\\' + v0, v0)
#                 else:
#                     try:
#                         value = float(value_part)
#                     except ValueError:
#                         value = value_part
#
#                 # word operators need spaces after them in the filter string,
#                 # but we don't want these later
#                 return name, operator_type[0].strip(), value
#
#     return [None] * 3
#
#
# if __name__ == '__main__':
#     app.run_server(debug=False)
