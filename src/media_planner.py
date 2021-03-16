import streamlit as st
from streamlit.report_thread import get_report_ctx
import pandas as pd
import base64
import os
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
import os
import sys


from opt_media import run_optimization
from SessionClass import _get_state
import PacingTools as pacing
# from PacingTools import pacing_calculation



matplotlib.use('Agg')

pd.options.mode.chained_assignment = None  # default='warn'


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_data(mydata):
    """
    docstring
    """

    extension = mydata.name.split('.')[-1]
    # st.write(extension)
    if extension == 'csv':
        df = pd.read_csv(mydata)
    elif extension == 'xlsx':
        df = pd.read_excel(mydata, engine='openpyxl')

    default_bounds = df[['Medias', 'MinInvestment', 'MaxInvestment']].copy()

    return df, default_bounds

def print_currency_dataframe(df, cols, currency_cols, percent_cols, object):
    formatted_df = df[cols]

    # currency_cols = ['CPM','CPC','CPA','MinInvestment','MaxInvestment']
    for col_name in currency_cols:
        formatted_df = format_column_currency(df=formatted_df, col_name=col_name)

    for col_name in percent_cols:
        formatted_df = format_column_percent(df=formatted_df, col_name=col_name)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(formatted_df.columns),
                    fill_color='gray',
                    font=dict(color='white', size=12),
                    align=['center',
                           'center',
                           'center',
                           'center',
                           'center',
                           'center'
                           ]),

        # cells_value = [ for ]

        cells=dict(
            values=formatted_df.values.T,
            fill_color='lavender',
            align=['center',
                   'right',
                   'right',
                   'right',
                   'right',
                   'right'
                   ]),
        # columnwidth=200
        # columnwidth = [250,250,250,250,250,250]
    )
    ])

    rows = formatted_df.shape[0]
    fig.update_layout(height=30 * rows,
                      margin=dict(t=0, b=0, l=0, r=0),
                      )

    element = object.plotly_chart(fig)

    return element

def print_st_dataframe(df, index_col, cols, currency_cols, percent_cols, df_object_source):
    formatted_df = df[cols]
    # formatted_df.set_index('Medias', inplace=True)
    for col_name in currency_cols:
        formatted_df = format_column_currency(df=formatted_df, col_name=col_name)

    for col_name in percent_cols:
        formatted_df = format_column_percent(df=formatted_df, col_name=col_name)

    formatted_df.sort_values(by=[index_col], ascending=True, inplace=True)
    formatted_df.reset_index(inplace=True, drop=True)
    df_object = df_object_source.table(formatted_df.set_index(index_col))
    # df_object.sort_values(by=['index_col'], ascending=True)

    # local_css('style.css')

    return df_object

def get_table_download_link(placeholder):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """

    dd = [
        {'Medias': 'media_1',
         'CPM': 0.01255,
         'CPC': 997.07,
         'CPA': 56.16,
         'MinInvestment': 600.0,
         'MaxInvestment': 2000.00},
        {'Medias': 'media_2',
         'CPM': 0.02094,
         'CPC': 1.2086299999999999,
         'CPA': 78.17,
         'MinInvestment': 900.0,
         'MaxInvestment': 3000.00},
        {'Medias': 'media_3',
         'CPM': 0.01744,
         'CPC': 5.59515,
         'CPA': 85.68,
         'MinInvestment': 450.0,
         'MaxInvestment': 1500.00},
        {'Medias': 'media_4',
         'CPM': 0.02437,
         'CPC': 1.54027,
         'CPA': 97.17,
         'MinInvestment': 600.0,
         'MaxInvestment': 2000.00},
        {'Medias': 'media_5',
         'CPM': 0.01961,
         'CPC': 5.19696,
         'CPA': 87.97,
         'MinInvestment': 900.0,
         'MaxInvestment': 3000.00}]

    df = pd.DataFrame(dd)
    df.to_csv('data_template.csv', index=False)

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    # href = f'<a href="data:file/csv;base64,{b64}" download="data_template.csv">Download csv file</a>'
    href = f'''<div class="row-widget stButton">
                    <button kind="primary" class="css-2trqyj edgvbvh1">
                    <a href="data:file/csv;base64,{b64}" download="data_template.csv">Download</a>
                    </button>
                    </div>'''
    # return href
    href = f'''<div class="row-widget stButton">
                        <a href="data:file/csv;base64,{b64}" download="data_template.csv">
                        <button kind="primary" class="css-2trqyj edgvbvh1">Download</button>
                        </a>    
                        </div>'''

    st.markdown(href, unsafe_allow_html=True)
    placeholder.empty()

def write_title_h2(title):
    return st.markdown(''' 
    <div style="text-align:center; font-weight: bold">
    <h2 class='step'>''' + title + '''</h2> 
    </div> ''', unsafe_allow_html=True)

def format_column_currency(df, col_name):
    # df[col_name] = df[col_name].apply(lambda x: 'R$ ' + str(x).replace('.', ','))
    df[col_name] = df[col_name].apply(lambda x: str(x).replace('.', ','))
    df[col_name] = df[col_name].apply(lambda x: str("{:,}".format(int(x.split(',')[0]))) + '|' + x.split(',')[1])
    df[col_name] = df[col_name].apply(lambda x: x.replace(',', '.').replace('|', ','))
    df[col_name] = df[col_name].apply(lambda x: 'R$ ' + str(x))
    return df

def format_column_percent(df, col_name):
    df[col_name] = df[col_name].apply(lambda x: str((x * 100)).replace('.', ','))
    df[col_name] = df[col_name].apply(lambda x: x.split(',')[0] + ',' + x.split(',')[1][0:1] + '%')
    # df[col_name] = df[col_name].apply(lambda x:  str((x*100)).replace('.', ',') + '%')
    return df

def create_page_navbar(title):
    navbar_fixed = '''
    <div class="navbar">''' + title + '''</div>
    '''
    st.markdown(navbar_fixed, unsafe_allow_html=True)

def add_media(params):


    st.markdown('''<style>
    #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(10){
    display: none !important;
    }
    </style>''', unsafe_allow_html=True)

    drop_columns = ['Optimal Investment', 'Share of Investment',
                    'Expected Clicks',
                    'Share of Clicks',
                    'Expected Impressions',
                    'Share of Impressions',
                    'Expected Acquisition',
                    'Share of Acquisition',
                    ]

    temp_df = params['state'].df
    for drop_col in drop_columns:
        if drop_col in temp_df:
            temp_df.drop(columns=[drop_col], axis=1, inplace=True)

    # variaveis = st.beta_columns(params['state'].df.shape[1] + 1)
    # column_names = params['state'].df.columns

    # variaveis = st.beta_columns(temp_df.shape[1] + 1)
    variaveis = st.beta_columns(temp_df.shape[1] + 1)
    col_vector = [2] * temp_df.shape[1]
    col_vector.append(1)
    variaveis = st.beta_columns(col_vector)

    column_names = temp_df.columns

    record = {}
    existing_media_warning = False
    for index_var, col in enumerate(variaveis):
        if index_var == 0:
            with col:
                media_text_placeholder = st.empty()
                col_name = column_names[index_var]
                new_media = media_text_placeholder.text_input(label=col_name, key='new_media', value='')
                index, = np.where(params['state'].media == new_media)
                record[col_name] = new_media

        elif index_var > 0 and index_var < len(variaveis) - 1:
            with col:
                col_name = column_names[index_var]
                # if new_media in params['state'].media:
                #     # previous_value = params['df'][col_name][index[0]]
                #     previous_value = params['state'].df[col_name][index[0]]

                new_val = st.number_input(col_name, step=0.00001, value=0.0, format='%g', key='New_{}'.format(col_name))
                record[col_name] = new_val

            # elif index_var == len(variaveis)-1 :

    with variaveis[-1]:
        # save_record = st.checkbox("Save Record")
        save_record = st.button("Add Media")
        if save_record:


            params['state'].show_add_media = 0


            if index.shape[0] > 0 :
                existing_media_warning = 1
            elif new_media.strip() == "":
                existing_media_warning = 2
            elif index.shape[0] == 0 and new_media.strip() > "":

                if params['state'].new_channels is None:
                    params['state'].new_channels = []

                params['state'].new_channels.append(record)

                # new_df = params['df'].append(params['state'].new_channels, ignore_index=True)
                new_df = params['state'].df.append(params['state'].new_channels, ignore_index=True)

                new_df.reset_index(inplace=True, drop=True)
                new_df.drop_duplicates(subset=['Medias'], inplace=True, keep='last')
                new_df.reset_index(inplace=True, drop=True)

                media = new_df.Medias.values
                default_bounds = new_df[['Medias', 'MinInvestment', 'MaxInvestment']].copy()

                # params['df'] = new_df.copy()
                # params['media'] = media
                # params['default_bounds'] = default_bounds

                params['state'].df = new_df.copy()
                params['state'].df.sort_values(by=['Medias'], ascending=True, inplace=True)
                params['state'].df.reset_index(inplace=True, drop=True)
                params['state'].media = media.copy()
                params['state'].media = np.sort(params['state'].media)
                params['state'].default_bounds = default_bounds.copy()


                st.markdown('''<style>
                #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(8){
                display: none !important;
                }
                </style>''', unsafe_allow_html=True)

                st.markdown('''<style>
                #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(10){
                display: none !important;
                }
                </style>''', unsafe_allow_html=True)






    if existing_media_warning == 1:
        st.warning("Media already exists!")
    elif existing_media_warning == 2:
        st.warning("Media needs to be filled!")

    planner = print_st_dataframe(
            df=params['state'].df,
            index_col='Medias',
            cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
            currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
            percent_cols=[],
            df_object_source=params['planner_obj'])

    params['planner_obj'] = planner

    return params

def edit_media(params):
    # with st.beta_expander('Edit Media', expanded=True):



    drop_columns = ['Optimal Investment', 'Share of Investment',
                    'Expected Clicks',
                    'Share of Clicks',
                    'Expected Impressions',
                    'Share of Impressions',
                    'Expected Acquisition',
                    'Share of Acquisition',
                    ]

    temp_df = params['state'].df
    for drop_col in drop_columns:
        if drop_col in temp_df:
            temp_df.drop(columns=[drop_col], axis=1, inplace=True)

    # variaveis = st.beta_columns(params['state'].df.shape[1] + 1)
    # column_names = params['state'].df.columns


    variaveis = st.beta_columns(temp_df.shape[1] + 2)


    col_vector = [2] * temp_df.shape[1]
    col_vector.append(1)
    col_vector.append(1)
    variaveis = st.beta_columns(col_vector)

    column_names = temp_df.columns

    record = {}
    for index_var, col in enumerate(variaveis):
        if index_var == 0:
            with col:
                media_text_placeholder = st.empty()
                # edit_media = media_text_placeholder.text_input("Medias")

                if params['state'].edit_index >= 0:
                    pass
                else:
                    params['state'].edit_index = 0

                # edit_media = st.selectbox('Choose Media', params['state'].media,
                #                           index=params['state'].edit_index, key='choose_media')


                edit_media = st.text_input('Medias', value=params['state'].edit_media)

                # index, = np.where(params['media'] == new_media)
                index, = np.where(params['state'].media == edit_media)
                previous_value = 0.0
                col_name = column_names[index_var]
                record[col_name] = edit_media


        # elif index_var >0 and index_var < len(variaveis)-1:
        elif index_var > 0 and index_var < len(variaveis) - 2:
            with col:
                col_name = column_names[index_var]
                # if new_media in params['media']:
                if edit_media in params['state'].media:
                    # previous_value = params['df'][col_name][index[0]]
                    previous_value = params['state'].df[col_name][index[0]]
                new_val = st.number_input(col_name, step=0.00001, value=previous_value, format='%g',key='Edit_{}'.format(col_name))
                record[col_name] = new_val

            # elif index_var == len(variaveis)-1 :

    with variaveis[-2]:
        # save_record = st.checkbox("Save Record")
        save_record = st.button("Save Record")
        if save_record:

            if params['state'].new_channels is None:
                params['state'].new_channels = []

            params['state'].new_channels.append(record)

            # new_df = params['df'].append(params['state'].new_channels, ignore_index=True)
            new_df = params['state'].df.append(params['state'].new_channels, ignore_index=True)

            new_df.reset_index(inplace=True, drop=True)
            new_df.drop_duplicates(subset=['Medias'], inplace=True, keep='last')
            new_df.reset_index(inplace=True, drop=True)

            media = new_df.Medias.values
            default_bounds = new_df[['Medias', 'MinInvestment', 'MaxInvestment']].copy()

            # params['df'] = new_df.copy()
            # params['media'] = media
            params['default_bounds'] = default_bounds

            params['state'].df = new_df.copy()
            params['state'].df.sort_values(by=['Medias'], ascending=True, inplace=True)
            params['state'].df.reset_index(inplace=True, drop=True)
            params['state'].media = media.copy()
            params['state'].media = np.sort(params['state'].media)
            params['state'].default_bounds = default_bounds.copy()


            st.markdown(
            '''<style>
            #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(8){
            display: none !important;
            }
            </style>''', unsafe_allow_html=True)

            st.markdown(
            '''<style>
            #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(10){
            display: none !important;
            }
            </style>''', unsafe_allow_html=True)

            output = run_optimization(params)
            params['state'].output = output


    # with variaveis[-2]:
    #     delete_media = st.selectbox('Choose Media', params['state'].media)

    with variaveis[-1]:

        confirm_delete = st.button("Delete")
        if confirm_delete:
            params['state'].df = params['state'].df[params['state'].df['Medias'] != edit_media]

            params['state'].df.reset_index(inplace=True, drop=True)

            params['state'].media = params['state'].df.Medias.values
            params['state'].media = np.sort(params['state'].media)

            # new_media = media_text_placeholder.text_input("Medias", value="")
            new_media = ""
            index, = np.where(params['state'].media == new_media)
            previous_value = 0.0

            st.markdown(
            '''<style>
            #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(8){
            display: none !important;
            }
            </style>''', unsafe_allow_html=True)

            st.markdown(
            '''<style>
            #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(10){
            display: none !important;
            }
            </style>''', unsafe_allow_html=True)

            output = run_optimization(params)
            params['state'].output = output


    planner = print_st_dataframe(
        df=params['state'].df,
        index_col='Medias',
        cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
        currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
        percent_cols=[],
        df_object_source=params['planner_obj'])

    params['planner_obj'] = planner


    return params

def create_app_header():
    input_1_col, input_2_col, input_3_col, input_4_col, input_5_col, input_6_col = st.beta_columns([1, 2, 1, 1, 2, 1])

    data = None
    df = None
    default_bounds = None
    min_lower_number = None
    Budget = None
    objective = None
    duration = None
    spend_all = None
    target = None
    media = None
    spend_bounds = None
    optimize_button = None
    campaign_duration = None
    campaign_pacing = None
    planner_obj = st.empty()
    new_media_checkbox_obj = st.empty()
    ctx = get_report_ctx()
    session_id = ctx.session_id
    state = _get_state()

    with input_1_col:
        step_1 = st.beta_expander("Step 1", expanded=True)
        with step_1:
            st.markdown('''<label class="css-145kmo2 effi0qh0">Template File</label>''',
                        unsafe_allow_html=True)

            placeholder = st.empty()
            get_table_download_link(placeholder)

    with input_2_col:
        step_2 = st.beta_expander("Step 2", expanded=True)
        with step_2:
            data = st.file_uploader('Upload media file: XLSX or CSV', type=['xlsx', 'csv'], key='data')
            warnings = False
            warning_low_greater_than_upper = False

            if data is not None:
                df, default_bounds = load_data(data)
                df.sort_values(by=['Medias'], ascending = True, inplace=True)
                df.reset_index(inplace=True, drop=True)
                # df, default_bounds = load_data_2(data, session_id=session_id)

                cols = df.columns
                for c in cols:
                    if c not in 'Medias':
                        df[c] = df[c].astype(float)

                min_lower_number = df['MinInvestment'].min()

    if data is not None:
        with input_3_col:
            with st.beta_expander("Step 3", expanded=True):
                Budget = st.number_input('Input budget limit:', value=default_bounds.MaxInvestment.sum(), key='budget')

        with input_4_col:
            with st.beta_expander("Step 4", expanded=True):
                # objective = st.selectbox('Campaign Objective:',
                #                          ['Clicks', 'Impressions', 'Acquisition']  # ,'Conversion']
                #                          , key='objective')

                campaign_duration = st.number_input("Campaign Duration", step=1.0, value=15.0, key='duration')

                # duration = st.number_input("Input Campaign Duration", step=1.0)

        spend_all = True

        target = {
            'Clicks': 'CPC',
            'Impressions': 'CPM',
            'Acquisition': 'CPA',
            # 'Conversion':'Conversion rate',
        }

        # Budget = 10000.00
        media = df.Medias.values

        MinInvest = default_bounds[['Medias', 'MinInvestment']].set_index('Medias').to_dict()['MinInvestment']
        MaxInvest = default_bounds[['Medias', 'MaxInvestment']].set_index('Medias').to_dict()['MaxInvestment']

        with input_5_col:
            step_5 = st.beta_expander("Step 5", expanded=True)
            with step_5:
                # spend_bounds = st.checkbox('Custom spending bounds', value=False, key='spending_bounds_check')
                campaign_pacing = st.selectbox('Campaign Pacing:',
                                         ['Uniform', 'Increasing', 'Decreasing',
                                          'V_Shaped', 'Inverted_V_Shaped']  # ,'Conversion']
                                         , key='pacing')

        with input_6_col:
            step_6 = st.beta_expander("Step 6", expanded=True)
            with step_6:
                st.markdown('''<label class="css-145kmo2 effi0qh0">Optimal Media Plan</label>''',
                            unsafe_allow_html=True)
                optimize_button = st.button("Run Optimization", key='optimize_button')

    params = {}
    params['data'] = data
    params['warnings'] = warnings
    params['warning_low_greater_than_upper'] = warning_low_greater_than_upper
    params['df'] = df
    params['default_bounds'] = default_bounds
    params['min_lower_number'] = min_lower_number
    params['Budget'] = Budget
    params['objective'] = objective
    params['duration'] = duration
    params['spend_all'] = spend_all
    params['target'] = target
    params['media'] = media
    params['spend_bounds'] = spend_bounds
    params['optimize_button'] = optimize_button
    params['planner_obj'] = st
    params['new_media_checkbox_obj'] = new_media_checkbox_obj
    params['session_id'] = session_id
    params['state'] = state
    params['duration'] = campaign_duration
    params['pacing'] = campaign_pacing

    if params['state'].df is None:
        params['state'].df = df
    if params['state'].media is None:
        params['state'].media = media
    if params['state'].default_bounds is None:
        params['state'].default_bounds = default_bounds

    # params['state'].show_add_media = False
    # params['state'].df = params['df']

    # if new_media_checkbox == True:
    #     planner, params = add_new_media(new_media_checkbox=new_media_checkbox, params=params, planner=st)

    return params

def print_st_button(mchannel, df_object_source):
    df_object = df_object_source.button(
        label="Edit {}".format(mchannel), key='edit_{}'.format(mchannel))
    # df_object = df_object_source.button(
    #     label="Edit {}".format(mchannel))
    return df_object

def display_planner_container(params):

    params['spend_bounds'] = False


    params['state'].media = np.sort(params['state'].media)

    for mchannel in params['state'].media:
        # index, = np.where(params['media'] == mchannel)
        index, = np.where(params['state'].media == mchannel)


    with st.beta_expander('Planner', expanded=True):


        col_planner, col_edit = st.beta_columns([9,1])
        with col_planner:

            params['state'].df.sort_values(by=['Medias'], ascending=True, inplace=True)
            params['state'].df.reset_index(inplace=True, drop=True)
            planner = print_st_dataframe(
                df=params['state'].df,
                index_col='Medias',
                cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                percent_cols=[],
                df_object_source=params['planner_obj'])

            params['planner_obj'] = planner

        with col_edit:


            for mchannel in params['state'].media:

                edit_btn = st.button(label="Edit {}".format(mchannel), key='edit_{}'.format(mchannel))

                index, = np.where(params['state'].media == mchannel)
                if edit_btn:
                    params['state'].show_edit_media = True
                    params['state'].show_add_media = False
                    params['state'].edit_index = index[0].item()
                    params['state'].edit_media = mchannel

            # if edit_btn:
            #     params['state'].show_edit_media = True
            #     params['state'].show_add_media = False
            #     params['state'].edit_index = index[0].item()
            #     params['state'].edit_media = mchannel


        add_media_button = st.button(label="", key='add_media_button')
        if add_media_button:
            # params['state'].show_add_media = True
            params['state'].show_edit_media = False

            if params['state'].show_add_media:
                params['state'].show_add_media = False
            else:
                params['state'].show_add_media = True

            # st.markdown('''<style>
            # #root>div:nth-child(1)>div>div>div>div>section>div>div:nth-child(1)>div:nth-child(10){
            # display: none !important;
            # }
            # </style>''', unsafe_allow_html=True)

    if params['state'].show_add_media:
        params = add_media(params)

    # with st.beta_expander('Add Media', expanded=False):
    #     params = add_media(params)

    if params['state'].show_edit_media:
        params = edit_media(params)




    # with st.beta_expander('Edit / Delete Media', expanded=False):
    #     params = edit_media(params)


    # new_media_checkbox = st.checkbox("Edit Media", key="Edit Media")
    # if new_media_checkbox:
    #     # record = add_new_media(params)
    #     params = add_new_media(params)


    return params

def display_optimization_plan(params):
    if params['Budget'] <= 0:
        params['warnings'] = True

    if params['Budget'] < params['min_lower_number']:
        params['warnings'] = True

    if params['optimize_button']:
        params['state'].show_optimization = True


    if params['optimize_button'] and params['warnings'] == True:
        if params['Budget'] <= 0:
            st.warning("Input Budget must be greater than zero")
        if params['Budget'] <= params['min_lower_number']:
            st.warning("Input Budget must be greater than at least one of the Min Investments")
        if params['warning_low_greater_than_upper']:
            st.warning("Min Investment must be lower than Max Investment")
    elif params['optimize_button'] and params['warnings'] == False:

        output = run_optimization(params)
        params['state'].output = output


    if params['state'].show_optimization:

        with st.beta_expander("Media Plan - Recomendations", expanded=True):
            # output = run_optimization(params)
            # x,status = run_optimization(params)

            objectives = list(params['state'].output.keys())
            all_objectives = objectives.copy()
            all_objectives.remove('Multi-Objective')
            pd_dict = {}

            try:
                for objective in objectives:

                    x = params['state'].output[objective][0]
                    status = params['state'].output[objective][1]

                    if status == 'Optimal':


                        main_objective = objective

                        params['state'].df['Optimal Investment'] = params['state'].df.Medias.apply(
                            lambda j: x[j].value())
                        total_investiment = params['state'].df['Optimal Investment'].sum()
                        params['state'].df['Share of Investment'] = params['state'].df[
                                                                        'Optimal Investment'] / total_investiment

                        for objective in all_objectives:
                            # params['state'].df['Optimal Investment'] = params['state'].df.Medias.apply(
                            #     lambda j: x[j].value())
                            params['state'].df['Expected {}'.format(objective)] = round(
                                params['state'].df['Optimal Investment'] / params['state'].df[params['target'][objective]],
                                0)

                            total_objective = params['state'].df['Expected {}'.format(objective)].sum()
                            params['state'].df['Share of {}'.format(objective)] = params['state'].df['Expected {}'.format(
                                objective)] / total_objective

                        cols = ['Medias', 'Optimal Investment', 'Share of Investment']
                        for objective in all_objectives:
                            cols.append('Expected {}'.format(objective))
                            cols.append('Share of {}'.format(objective))

                        percent_cols = ['Share of Investment']
                        for objective in all_objectives:
                            percent_cols.append('Share of {}'.format(objective))

                        pd_dict[main_objective] = params['state'].df



                        # with st.beta_expander("Optimal Media Plan - {}".format(main_objective), expanded=False):
                        optmization_df_ch = st.checkbox(label='Target - {}'.format(main_objective), key=main_objective)
                        if optmization_df_ch:
                            optimal = print_st_dataframe(
                                df=params['state'].df,
                                index_col='Medias',
                                cols=cols,
                                currency_cols=['Optimal Investment'],
                                percent_cols=percent_cols,
                                df_object_source=st)

                            csv = params['state'].df[cols].to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                            href = f'<a href="data:file/csv;base64,{b64}" download="optimal_investment_' + main_objective + '.csv">Export optimal media plan as CSV</a> (right-click and save as &lt;some_name&gt;.csv)'
                            st.markdown(href, unsafe_allow_html=True)


                    elif status == 'Infeasible':
                        st.error(
                            'Infeasible! Adjust budget and try again'
                        )

            except KeyError:
                st.warning(
                    'New Media was added. Please Run Optimization again'
                )



            if pd_dict:
                params['state'].optimization = pd_dict

    return params


def plot_pacing(params):


    params['df_pacing'] = params['state'].pacing_dict['Clicks']

    with st.beta_expander(label='Pacing {}'.format(params['pacing']), expanded=True):
        for objective in params['state'].pacing_dict.keys():
            # objective = 'Clicks'
            # with st.beta_expander(label='Pacing {} - Target {}'.format(params['pacing'], objective), expanded=False):
            # with st.beta_expander(label='Pacing {}'.format(params['pacing']), expanded=False):
            params['df_pacing'] = params['state'].pacing_dict[objective]
            pacing_df_ch = st.checkbox(label='Target - {}'.format(objective), key=''.format(objective))
            if pacing_df_ch:


                # st.table(df_planner)
                cols_medias = params['df_pacing'].columns.tolist()
                cols_medias.remove('day')
                df_pacing = params['df_pacing']

                df_new = pd.melt(df_pacing, id_vars=['day'], value_vars=cols_medias,
                                 var_name='Medias', value_name='Investment')

                # st.table(df_new)

                fig = px.line(df_new,
                              x='day',
                              y='Investment',
                              color='Medias',
                              )
                fig.update_layout(
                    height=300,
                    width=1100,
                    xaxis_title='Day',
                    yaxis_title='Investment',
                    # legend=dict(
                    #     yanchor="top",
                    #     y=-1.10,
                    #     xanchor="right",
                    #     x=0.80,
                    #     font=dict(family="Courier", size=10, color="black")
                    # ),
                    hovermode='x'
                )

                # fig = px.line(df_new,
                #               x='day',
                #               y='Investment',
                #               color='Medias',
                #               barmode="overlay")

                st.plotly_chart(fig)

                st.table(params['df_pacing'].set_index('day'))



def main():
    st.set_page_config(page_title='MIP - Media Investment Planner',
                       page_icon=':card_file_box:',
                       layout='wide',
                       initial_sidebar_state='collapsed')

    create_page_navbar(title='MIP - Media Investment Planner')

    local_css(os.path.join(os.getcwd(), 'src', 'style.css'))

    # st.markdown('<i class="material-icons">face</i>', unsafe_allow_html=True)

    params = create_app_header()

    if params['data'] is not None:
        params = display_planner_container(params)
        params = display_optimization_plan(params)
        # params = pacing.campaign_pacing(params)
        if params['state'].optimization is not None:
            params = pacing.pacing_calculation(params)
            # params = pacing_calculation(params)
            params = plot_pacing(params)





if __name__ == "__main__":
    main()

