import pandas as pd
import numpy as np

def pre_process_table(table):
    """
    INPUT:
        1. table - a bs4 element that contains the desired table: ie <table> ... </table>
    OUTPUT:
        a tuple of:
            1. rows - a list of table rows ie: list of <tr>...</tr> elements
            2. num_rows - number of rows in the table
            3. num_cols - number of columns in the table
    Options:
        include_td_head_count - whether to use only th or th and td to count number of columns (default: False)
    """
    rows = [x for x in table.find_all('tr')]

    num_rows = len(rows)

    # get an initial column count. Most often, this will be accurate
    num_cols = max([len(x.find_all(['th','td'])) for x in rows])

    # sometimes, the tables also contain multi-colspan headers. This accounts for that:
    header_rows_set = [x.find_all(['th', 'td']) for x in rows if len(x.find_all(['th', 'td']))>num_cols/2]

    num_cols_set = []

    for header_rows in header_rows_set:
        num_cols = 0
        for cell in header_rows:
            row_span, col_span = get_spans(cell)
            num_cols+=len([cell.get_text(strip=True)]*col_span)

        num_cols_set.append(num_cols)

    num_cols = max(num_cols_set)

    return (rows, num_rows, num_cols)


def get_spans(cell):
        """
        INPUT:
            1. cell - a <td>...</td> or <th>...</th> element that contains a table cell entry
        OUTPUT:
            1. a tuple with the cell's row and col spans
        """
        if cell.has_attr('rowspan'):
            rep_row = int(cell.attrs['rowspan'])
        else: # ~cell.has_attr('rowspan'):
            rep_row = 1
        if cell.has_attr('colspan'):
            rep_col = int(cell.attrs['colspan'])
        else: # ~cell.has_attr('colspan'):
            rep_col = 1

        return (rep_row, rep_col)

def process_rows(page, rows, num_rows, num_cols):
    links, dates = [], []
    """
    INPUT:
        1. rows - a list of table rows ie <tr>...</tr> elements
    OUTPUT:
        1. data - a Pandas dataframe with the html data in it
    """
    data = pd.DataFrame(np.ones((num_rows+1, num_cols))*np.nan)
    reprows = 0
    for i, row in enumerate(rows):
        got_sup = 0
        try:
            col_stat = data.iloc[i,:][data.iloc[i,:].isnull()].index[0]
        except IndexError:
            print(i, row)

        # dates += ['05 March 2020']
        # links += ['b']

        for j, cell in enumerate(row.find_all(['td', 'th'])):
            rep_row, rep_col = get_spans(cell)

            if not got_sup:
                sup = cell.find('sup')
                if sup is not None:
                    cite_id = sup.find('a')['href'][1:]
                    cite_elem = page.find(id=cite_id)
                    link, date = '', pd.datetime.now().strftime('%d %B %Y')
                    if cite_elem is not None:
                        link_elem = cite_elem.find('a', {"class": 'external text'})
                        date_elem = cite_elem.find('span', {"class": "reference-accessdate"})
                        if link_elem is not None and date_elem is not None:
                            link = link_elem['href'].strip()
                            date = date_elem.text[12:] if date_elem is not None else None
                    dates += [date] * rep_row
                    links += [link] * rep_row
                    reprows += rep_row
                    got_sup += 1

            for sup in cell.findAll('sup'):
                sup.decompose()

            #print("cols {0} to {1} with rep_col={2}".format(col_stat, col_stat+rep_col, rep_col))
            #print("\trows {0} to {1} with rep_row={2}".format(i, i+rep_row, rep_row))

            #find first non-na col and fill that one
            while any(data.iloc[i,col_stat:col_stat+rep_col].notnull()):
                col_stat+=1

            data.iloc[i:i+rep_row,col_stat:col_stat+rep_col] = cell.get_text(strip=True)
            if col_stat<data.shape[1]-1:
                col_stat+=rep_col

    print(num_rows, reprows)

    return data, links, dates

def get_table(table, page):
    rows, num_rows, num_cols = pre_process_table(table)
    return process_rows(page, rows, num_rows, num_cols)