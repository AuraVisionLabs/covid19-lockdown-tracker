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

with open('areweinlockdown-com/dist/countrylist.json') as f:
  js_arewe = json.load(f)
df_arewe = pd.DataFrame([{
    'Country': r['name'], 'Lockdown': (r['situation']['state'] == 'YES'), 'Shops': ('essential' in r['situation']['shop']) if 'shop' in r['situation'].keys() else None
} for r in js_arewe])
df_arewe_ls = df_arewe[df_arewe['Lockdown'] | df_arewe['Shops']]

wiki_inter_req = requests.get("https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic")
wiki_inter_soup = BeautifulSoup(wiki_inter_req.content, "html.parser")
wiki_inter_table = wiki_inter_soup.find_all("span", string="Coronavirus quarantines outside China")[0].find_parent('table')

wiki_china_req = requests.get("https://en.wikipedia.org/wiki/2020_Hubei_lockdowns")
wiki_china_soup = BeautifulSoup(wiki_china_req.content, "html.parser")
wiki_china_table = wiki_china_soup.find_all("span", string="Cities under quarantine in China")[0].find_parent('table')
wiki_china_table.find_all("th", text=re.compile('^Quarantine total$'))[0].find_parent('tr').decompose()
wiki_china_table.findAll('tr')[-1].decompose()

def remove_sup(tab, soup):
    links = []
    dates = []
    row_list = tab.findAll('tr')
    for tr in row_list:
        sup = tr.find('sup')
        if sup is not None:
            cite_id = sup.find('a')['href'][1:]
            cite_elem = soup.find(id=cite_id)
            link = cite_elem.find('a', {"class": 'external text'})['href']
            date_elem = cite_elem.find('span', {"class": "reference-accessdate"})
            date = date_elem.text[12:] if date_elem is not None else None
            # print(link, date)
            dates.append(date)
            links.append(link)
            sup.decompose()
        else:
            tr.decompose()
    for sup in tab.findAll('sup'):
        sup.decompose()
    return links, dates


wiki_inter_links, wiki_inter_dates = remove_sup(wiki_inter_table, wiki_inter_soup)
wiki_china_links, wiki_china_dates = remove_sup(wiki_china_table, wiki_china_soup)

df_wiki_inter = pd.read_html(str(wiki_inter_table), header=None)[0]
print('wiki international', len(df_wiki_inter), len(wiki_inter_links))
df_wiki_inter.columns = ['Country', 'Place', 'Start date', 'End date', 'Level']
df_wiki_inter['url'] = wiki_inter_links
df_wiki_inter['update'] = pd.to_datetime(wiki_inter_dates, format='%d %B %Y')
df_wiki_inter['Confirmed'] = False
# print(df_wiki_inter)

df_wiki_china = pd.read_html(str(wiki_china_table), header=None)[0]
print('wiki china', len(df_wiki_china), len(wiki_china_links))
df_wiki_china = df_wiki_china.iloc[1:]
df_wiki_china = df_wiki_china[[0,2,3]]
df_wiki_china.columns = ['Place', 'Start date', 'End date']
df_wiki_china['url'] = wiki_china_links[1:]
df_wiki_china['update'] = pd.to_datetime(wiki_china_dates[1:], format='%d %B %Y')
df_wiki_china['Country'] = 'China'
df_wiki_china['Level'] = 'City'
df_wiki_china['Confirmed'] = True

df_wiki = pd.concat((df_wiki_inter, df_wiki_china), sort=False)
df_wiki.to_csv('wiki_lockdown_dates.csv', index=False)

df_aura = pd.read_csv('aura_lockdown_dates.csv')
df_aura['update'] = pd.to_datetime(df_aura['update'], format='%Y-%m-%d')

df_quar = pd.concat((df_wiki_inter, df_wiki_china, df_aura), sort=False)
df_quar['update'] = df_quar['update'].fillna(pd.Timestamp('2000/11/12 13:35'))
df_quar = df_quar.sort_values('update')
df_quar = df_quar.drop_duplicates(['Country', 'Place'], keep='last')
df_quar = df_quar.dropna(subset=['Start date'])
df_quar.loc[df_quar['Place'] == df_quar['Country'], 'Place'] = np.nan

print('not in arewe')
print(set(df_quar['Country']) - set(df_arewe_ls['Country']))
print('not in ours')
print(set(df_arewe_ls['Country']) - set(df_quar['Country']))

df_quar.to_csv('deploy/lockdown_dates.csv', index=False)
df_quar.to_csv('history/lockdown_dates_%s.csv' % (pd.datetime.now().strftime('%d-%m-%y')), index=False)
df_quar_old = pd.read_csv(
    'history/lockdown_dates_%s.csv' % ((pd.datetime.now() - pd.offsets.Day(1)).strftime('%d-%m-%y')))

df_old_new = pd.concat([df_quar, df_quar_old], sort=False)[['Country', 'Place', 'Start date', 'End date', 'update']]
df_old_new['update'] = pd.to_datetime(df_old_new['update'], format='%Y-%m-%d')
df_old_new = df_old_new.drop_duplicates(keep=False).sort_values(['Country', 'Place'])
print(df_old_new)

df_quar['Start'] = pd.to_datetime(df_quar['Start date'], format='%Y-%m-%d')
df_quar['Finish'] = pd.to_datetime(df_quar['End date'], format='%Y-%m-%d')

df_quar_china_us = df_quar[df_quar['Country'].isin(['United States', 'China'])]
df_quar_other = df_quar[~df_quar['Country'].isin(['United States', 'China'])]
df_quar_other = df_quar_other.groupby('Country').agg(
    {'Start': 'min', 'Finish': 'max', 'Place': lambda x: (str(len(x)) + ' places' if len(x) > 1 else x), 'Level': 'first', 'Confirmed': 'first', 'update': 'first', 'url': 'first'}).reset_index()
# print(df_quar_other)
df_quar = pd.concat((df_quar_china_us, df_quar_other), sort=False)

df_quar.loc[df_quar['Finish'] > (pd.datetime.now() + pd.offsets.Day(60)), 'Finish'] = np.nan
df_quar['Duration'] = df_quar['Finish'] - df_quar['Start']
df_quar['Name'] = np.where(df_quar['Level'] == 'National', df_quar['Country'],
                           df_quar['Country'] + ' (' + df_quar['Place'] + ')')
df_quar['CC'] = df_quar['Country'].apply(lambda x: pycountry.countries.search_fuzzy(x)[0].alpha_2)
df_quar['CCE'] = df_quar['CC'].apply(lambda x: flag.flag(x))
df_quar['Complete'] = df_quar['Finish'] <= pd.datetime.now()
print(df_quar)
df_quar = df_quar.sort_values(['Start', 'Country', 'Duration'], ascending=[False, False, True]).reset_index()

average_confirmed_days = df_quar[df_quar['Confirmed'] == True]['Duration'].mean().days
average_review_days = df_quar[df_quar['Confirmed'] == False]['Duration'].mean().days
max_confirmed_days = df_quar[df_quar['Confirmed']  == True]['Duration'].max().days
average_to_review_days = (df_quar[df_quar['Confirmed'] == False]['Finish'] - pd.datetime.now()).mean().days
average_into_lockdown_days = (pd.datetime.now() - df_quar[df_quar['Confirmed'] == False]['Start']).mean().days
total_finished_country = df_quar[(df_quar['Confirmed'] == True) & (df_quar['Complete'])]['Country'].nunique()
total_finished = len(df_quar[(df_quar['Confirmed'] == True) & (df_quar['Complete'])])
no_end = df_quar['Finish'].isna().sum()

x_points = [0, 1, 2, 3, 4]
x_text = ['<b>Lockdowns started</b><br>%d in %d Countries<br>%d Days ago️<br>on average' % (len(df_quar),
                                                                                            df_quar[
                                                                                                'Country'].nunique(),
                                                                                            average_into_lockdown_days),
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
        day_str = ('1 Day' if dtt * dtt == 1 else str(abs(dtt)) + ' Days')
        day_str = day_str + ' ago' if dtt < 0 else day_str + ' from now'
    update_str = '<br>Last updated on ' + row['update'].strftime('%d %b') if not pd.isna(row['update']) else ''
    return '%s %s - %s<br>%s - %s (%s)%s' % (
        row['CCE'], row['Name'], row['Level'], text, day_str, date.strftime('%d %b'), update_str)


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
             ) for index, row in df_quar[df_quar['Confirmed'] == False].iterrows()
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
             ) for index, row in df_quar[df_quar['Confirmed'] == False].iterrows()
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
                               ('Lockdown ended' if row['Finish'] <= pd.datetime.now() else 'Lockdown end confirmed') if
                               row['Confirmed'] else 'Lockdown review'),
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
                               'text': '<b>Last updated ' + pd.datetime.now().strftime(
                                   '%d %B %Y') + '</b><br>The most comprehensive source for how past and current lockdowns are unfolding, updated daily.<br>Lockdowns dates reflect when non-essential retail has been ordered to close by local government.<br>Interactive version and data download <a href="https://auravision.ai/covid19-lockdown-tracker">https://auravision.ai/covid19-lockdown-tracker</a>. Please share if you find this useful.',
                           },
                           {
                               'x': 1.0, 'y': 0, 'xref': 'paper', 'yref': 'paper',
                               'showarrow': False,
                               'font': {
                                   'size': 10,
                                   'color': 'rgb(39, 44, 63)'
                               },
                               'xanchor': 'right',
                               'yanchor': 'bottom',
                               'align': 'right',
                               'text': 'Major data sources include:<br><a href="https://areweinlockdown.com/all_countries.html">Are we in Lockdown?</a><br><a href="https://en.wikipedia.org/wiki/National_responses_to_the_2019%E2%80%9320_coronavirus_pandemic#In_other_countries">Wikipedia - National responses to the_pandemic</a>,<br><a href="https://www.businessinsider.com/countries-on-lockdown-coronavirus-italy-2020-3?r=US&IR=T">Business Insider - Countries on Lockdown</a><br><a href="https://edition.cnn.com/2020/03/23/us/coronavirus-which-states-stay-at-home-order-trnd/index.html">CNN - Which US States have stay-at-home order</a>.<br><b>Aura Vision is a provider of in-store retail analytics.<br>For more information please visit <a href="https://auravision.ai">https://auravision.ai</a></b>.'
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
            'font': {'size': 12},
            'xanchor': 'right',
            'yanchor': 'middle',

            'text': ('<b>' + row['Name'] + '</b>' if row['Level'] == 'National' else row[
                'Name']) + ' <span style="font-size: 18px">' + row['CCE'] + '  </span><a href="' + row['url'] + '">🔗</a> ' + row['Start'].strftime(
                '%d %b') + '  '
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
