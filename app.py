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
import json
from table_proc import get_table

with open('areweinlockdown-com/dist/countrylist.json') as f:
  js_arewe = json.load(f)
df_arewe = pd.DataFrame([{
    'Country': r['name'], 'Lockdown': (r['situation']['state'] == 'YES'), 'Shops': ('essential' in r['situation']['shop']) if 'shop' in r['situation'].keys() else None
} for r in js_arewe])
df_arewe_ls = df_arewe[df_arewe['Lockdown'] | df_arewe['Shops']]

# wiki_inter_req = requests.get("https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic")
# wiki_inter_soup = BeautifulSoup(wiki_inter_req.content, "html.parser")
# wiki_inter_table = wiki_inter_soup.find("span", string="2020 coronavirus pandemic lockdowns").find_parent('table')
# # wiki_inter_table.find_all("a", string="Blida").find_parent("tr").decompose()
# wiki_inter_table.findAll('tr')[0].decompose()
# wiki_inter_table.findAll('tr')[0].decompose()
# wiki_inter_table.findAll('tr')[-1].decompose()
#
# wiki_china_req = requests.get("https://en.wikipedia.org/wiki/2020_Hubei_lockdowns")
# wiki_china_soup = BeautifulSoup(wiki_china_req.content, "html.parser")
# wiki_china_table = wiki_china_soup.find("span", string="Cities under quarantine in China").find_parent('table')
# wiki_china_table.find_all("th", text=re.compile('^Quarantine total$'))[0].find_parent('tr').decompose()
# wiki_china_table.findAll('tr')[0].decompose()
# wiki_china_table.findAll('tr')[0].decompose()
# wiki_china_table.findAll('tr')[0].decompose()
# wiki_china_table.findAll('tr')[-1].decompose()
#
# df_wiki_inter, wiki_inter_links, wiki_inter_dates = get_table(wiki_inter_table, wiki_inter_soup)
# df_wiki_china, wiki_china_links, wiki_china_dates = get_table(wiki_china_table, wiki_china_soup)
#
# print('wiki international', len(df_wiki_inter), len(wiki_inter_links))
# # print(df_wiki_inter)
# df_wiki_inter.columns = ['Country', 'Place', 'Start date', 'End date', 'Level']
# df_wiki_inter['url'] = wiki_inter_links
# df_wiki_inter['update'] = pd.to_datetime(wiki_inter_dates, format='%d %B %Y')
# df_wiki_inter['Confirmed'] = False
# df_wiki_inter['Country'] = df_wiki_inter['Country'].str.replace(r'\([^)]*\)', '')
# # print(df_wiki_inter)
#
# df_wiki_china = pd.read_html(str(wiki_china_table), header=None)[0]
# print('wiki china', len(df_wiki_china), len(wiki_china_links))
# # print(df_wiki_china)
# df_wiki_china = df_wiki_china[[0,2,3]]
# df_wiki_china.columns = ['Place', 'Start date', 'End date']
# df_wiki_china['url'] = wiki_china_links
# df_wiki_china['update'] = pd.to_datetime(wiki_china_dates, format='%d %B %Y')
# df_wiki_china['Country'] = 'China'
# df_wiki_china['Level'] = 'City'
# df_wiki_china['Confirmed'] = True
#
# df_wiki = pd.concat((df_wiki_inter, df_wiki_china), sort=False)
# df_wiki.to_csv('wiki_lockdown_dates.csv', index=False)

df_aura = pd.read_csv('aura_lockdown_dates_sorted.csv')
df_aura.sort_values(['Country', 'Place']).to_csv('aura_lockdown_dates_sorted.csv', index=False)

df_aura['update'] = pd.to_datetime(df_aura['update'], format='%Y-%m-%d')

df_old_csv = pd.read_csv('history/lockdown_dates_%s.csv' % ((pd.datetime.now() - pd.offsets.Day(1)).strftime('%d-%m-%y')))
df_old_csv['update'] = pd.to_datetime(df_old_csv['update'], format='%Y-%m-%d')

df_quar = pd.concat((df_old_csv, df_aura), sort=False)
df_quar['update'] = df_quar['update'].fillna(pd.Timestamp('2000/11/12 13:35'))
df_quar.loc[df_quar['Place'] == df_quar['Country'], 'Place'] = np.nan
df_quar = df_quar.sort_values('update')
df_quar = df_quar.drop_duplicates(['Country', 'Place'], keep='last')
df_quar = df_quar.dropna(subset=['Start date'])

print('not in arewe')
print(set(df_quar['Country']) - set(df_arewe_ls['Country']))
print('not in ours')
print(set(df_arewe_ls['Country']) - set(df_quar['Country']))

df_quar.to_csv('deploy/lockdown_dates.csv', index=False)
df_quar.to_csv('history/lockdown_dates_%s.csv' % (pd.datetime.now().strftime('%d-%m-%y')), index=False)

def pre_proc_df(df):
    df['Start'] = pd.to_datetime(df['Start date'], format='%Y-%m-%d', errors='coerce')
    df['Finish'] = pd.to_datetime(df['End date'], format='%Y-%m-%d', errors='coerce')
    df = df.dropna(subset=['Start', 'Finish'])

    df_china_us = df[df['Country'].isin(['United States', 'China'])]
    df_other = df[~df['Country'].isin(['United States', 'China'])].sort_values(['Finish'], ascending=False)
    df_other = df_other.groupby('Country').agg(
        {'Start': 'min', 'Finish': 'max', 'Place': lambda x: (str(len(x)) + ' places' if len(x) > 1 else x), 'Level': 'first', 'Confirmed': 'first', 'update': 'first', 'url': 'first'}).reset_index()
    # print(df_other)
    df = pd.concat((df_china_us, df_other), sort=False)

    df.loc[df['Finish'] > (pd.datetime.now() + pd.offsets.Day(60)), 'Finish'] = np.nan
    df['Duration'] = df['Finish'] - df['Start']
    df['Name'] = np.where(df['Level'] == 'National', df['Country'],
                               df['Country'] + ' (' + df['Place'] + ')')
    df['CC'] = df['Country'].apply(lambda x: pycountry.countries.search_fuzzy(x)[0].alpha_2)
    df['CCE'] = df['CC'].apply(lambda x: flag.flag(x))
    df['Complete'] = df['Finish'] <= pd.datetime.now()
    df['update_status'] = ''
    # df = df.sort_values(['Start', 'Country', 'Duration'], ascending=[False, False, True]).reset_index()
    # df = df.sort_values(['Finish', 'Country', 'Duration'], ascending=[True, True, True]).reset_index()
    df = df.sort_values(['Country', 'Place', 'Duration'], ascending=[True, True, True]).reset_index()
    return df

df_quar = pre_proc_df(df_quar)
print(df_quar)
df_old = pre_proc_df(pd.read_csv('history/lockdown_dates_%s.csv' % ((pd.datetime.now() - pd.offsets.Day(1)).strftime('%d-%m-%y'))))
print(df_old)

for index, row in df_quar.iterrows():
    if row['Finish'] >= pd.datetime.now().date() and row['Confirmed']:
        df_quar.at[index, 'update_status'] = '✅ Stores reopening'
        continue

    if not row['Country'] in df_old['Country'].unique():
        df_quar.at[index, 'update_status'] = '📣 New country'
        continue

    old_row = df_old[(df_old['Country'] == row['Country']) & (df_old['Place'] == row['Place'])]
    if len(old_row) == 0:
        if not pd.isnull(row['Place']):
            df_quar.at[index, 'update_status'] = '📣 New place'
            continue
    else:
        old_row = old_row.iloc[0]
        # print(old_row)
        if pd.isnull(old_row['Finish']) and not pd.isnull(row['Finish']):
            df_quar.at[index, 'update_status'] = '⏰ Review date added'
            continue

        if old_row['Finish'] < row['Finish']:
            df_quar.at[index, 'update_status'] = '⏰ Review date updated'
            continue

average_confirmed_days = df_quar[df_quar['Confirmed'] == True]['Duration'].mean().days
average_review_days = df_quar[df_quar['Confirmed'] == False]['Duration'].mean().days
max_confirmed_days = df_quar[df_quar['Confirmed']  == True]['Duration'].max().days
average_to_review_days = (df_quar[df_quar['Confirmed'] == False]['Finish'] - pd.datetime.now()).mean().days
average_into_lockdown_days = (pd.datetime.now() - df_quar[df_quar['Confirmed'] == False]['Start']).mean().days
total_finished_country = df_quar[(df_quar['Confirmed'] == True) & (df_quar['Complete'])]['Country'].nunique()
total_finished = len(df_quar[(df_quar['Confirmed'] == True) & (df_quar['Complete'])])
no_end = df_quar['Finish'].isna().sum()

x_points = [0, 1, 2, 3, 4]
x_text = ['<b>Lockdowns started</b><br>%d in %d Countries' % (len(df_quar),
                                                                                            df_quar[
                                                                                                'Country'].nunique()),
          '<b>Lockdown reviews</b>',
          '<b>Non-essential retail opening</b><br>%d in %d Countries' % (total_finished, total_finished_country),
          '<b>Average duration</b><br>%d Days confirmed' % (average_confirmed_days),
          '<b>Maximum duration</b><br>%d Days confirmed' % (max_confirmed_days)]
x_color = ['rgb(255,0,90)', 'rgb(255,165,0)', 'rgb(0,255,165)', 'rgb(223,223,223)', 'rgb(180,180,180)']
line_height = -30
info_height = 0
offset_height = -100

# print(x_text, no_end)


def time_fmt(row, date, text, duration, update=None):
    dtt = (date - pd.datetime.now()).days + 1
    if dtt == 0:
        day_str = 'Today'
    else:
        day_str = ('1 Day' if dtt * dtt == 1 else str(abs(dtt)) + ' Days')
        day_str = day_str + ' ago' if dtt < 0 else day_str + ' from now'
    if update is None:
        update_str = ''
    else:
        update_str = '<br>Latest update - ' + row['update'].strftime('%d %b') if not pd.isna(update) else ''
    return '%s %s - %s<br>%s - %s (%s)<br>Duration - %s Days%s' % (
        row['CCE'], row['Name'], row['Level'], date.strftime('%d %b'), text, day_str, duration, update_str)


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
                 x=[row['Start'], row['Finish']],
                 y=[index * line_height + offset_height, index * line_height + offset_height],
                 mode='lines',
                 line={
                     'width': 3,
                     'color': x_color[2] if row['Finish'] <= pd.datetime.now() and row['Confirmed'] else
                     x_color[0],
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
                     'color': x_color[2] if row['Confirmed'] else x_color[1],
                 },
                 text=time_fmt(row, row['Finish'],
                               ('Non-essential retail opening' if row['Finish'] <= pd.datetime.now() else 'Non-essential retail opening') if
                               row['Confirmed'] else 'Lockdown review', row['Duration'].days, row['update']),
                 hoverinfo='text',
                 hoverlabel={
                     'bgcolor': 'white',
                     'bordercolor': x_color[2] if row['Confirmed'] else x_color[1],
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
                 text=time_fmt(row, row['Start'], 'Lockdown started', row['Duration'].days, row['update']),
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
                'size': 50,
                'color': 'rgb(49, 77, 160)'
            }

        },
        'hovermode': 'closest',
        'paper_bgcolor': 'rgba(255,255,255)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'height': 2350,
        'margin': {'t': 40, 'l': 0, 'r': 0, 'b': 0},
        'annotations': [
                           {
                               'x': 0.05, 'y': 0, 'xref': 'x3', 'yref': 'y3',
                               'font': {
                                   'size': 16,
                                   'color': 'rgb(39, 44, 63)'
                               },
                               'xanchor': 'left',
                               'yanchor': 'top',
                               'align': 'left',
                               'text': 'The most comprehensive source for how past and current lockdowns are unfolding.<br>Lockdowns dates reflect full or partial closure of non-essential retail, ordered by local government.<br>Interactive version and data download <a href="https://auravision.ai/covid19-lockdown-tracker">https://auravision.ai/covid19-lockdown-tracker</a>. Please share if you find this useful.',
                           },
                           {
                               'x': 0, 'y': 0, 'xref': 'paper', 'yref': 'paper',
                               'showarrow': False,
                               'font': {
                                   'size': 10,
                                   'color': 'rgb(39, 44, 63)'
                               },
                               'xanchor': 'left',
                               'yanchor': 'bottom',
                               'align': 'left',
                               'text': 'Major data sources include:<br><a href="https://www.nytimes.com/interactive/2020/us/states-reopen-map-coronavirus.html">NY Times - States Reopen Map</a><br><a href="https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic#In_other_countries">Wikipedia - National responses to the_pandemic</a>,<br><a href="https://www.businessinsider.com/countries-on-lockdown-coronavirus-italy-2020-3?r=US&IR=T">Business Insider - Countries on Lockdown</a><br><a href="https://edition.cnn.com/2020/03/23/us/coronavirus-which-states-stay-at-home-order-trnd/index.html">CNN - Which US States have stay-at-home order</a>.<br><b>Aura Vision is a provider of in-store retail analytics.<br>For more information please visit <a href="https://auravision.ai">https://auravision.ai</a></b>.'
                           }
                       ] + [
                           {
                               'x': x, 'y': 0, 'xref': 'x1', 'yref': 'y1', 'showarrow': True,
                               'arrowhead': 0, 'ax': 0, 'ay': -60, 'arrowcolor': color, 'arrowwidth': 2,
                               'font': {'size': 16},
                               'xanchor': 'center',
                               'text': text
                           } for x, text, color in zip(x_points, x_text, x_color)
                       ] + [{
            'x': pd.Timestamp(year=2019, month=12, day=14) if row['Country'] == 'China' else pd.Timestamp(year=2020, month=2, day=1), 'y': index * line_height + offset_height + 5, 'xref': 'x2', 'yref': 'y2',
            'showarrow': False,
            'font': {'size': 12},
            'xanchor': 'left',
            'yanchor': 'middle',
            'text': ('<b>' + row['Name'] + '</b>' if row['Level'] == 'National' else row['Name']) +
                    ' <span style="font-size: 18px">' + row['CCE'] + '  </span><a href="' + row['url'] + '">🔗</a> ' +
                    row['Start'].strftime('%d %b') + '  '
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
            'tickvals': [pd.datetime.now().date() + pd.offsets.Day(7 * (i - 50))
                         for i in
                         range(100)],
            'ticktext': [(pd.datetime.now().date() + pd.offsets.Day(7 * (i - 50))).strftime('%d %b') if i%2 == 0 else ''
                         for i in
                         range(100)],
            'tickfont': {
                'size': 16
            },
            'side': 'top',
            'gridcolor': 'rgb(240,240,240)',
            'fixedrange': True
        },
        'yaxis2': {
            'domain': [0, 0.85],
            'showgrid': False,
            'zeroline': True,
            'showticklabels': False,
            'range': [len(df_quar) * line_height + offset_height, 0],
            'zerolinewidth': 1,
            'zerolinecolor': 'rgb(90,90,90)',
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
