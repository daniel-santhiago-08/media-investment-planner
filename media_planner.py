import streamlit as st
import pandas as pd
import pulp as plp
import base64
from io import BytesIO 
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


def run_optimization(media, target, objective, Budget, spend_all, df):

    coef = df[['Medias',target[objective]]].set_index('Medias').to_dict()[target[objective]]
    MinInvest = df[['Medias','MinInvestment']].set_index('Medias').to_dict()['MinInvestment']
    MaxInvest = df[['Medias','MaxInvestment']].set_index('Medias').to_dict()['MaxInvestment']

    bigM = {i:MaxInvest[i]+1 for i in media}

    prob = plp.LpProblem("media_plan", plp.LpMaximize) 

    x = plp.LpVariable.dicts("x", (media), lowBound=0, cat='Continuous')
    y = plp.LpVariable.dicts("y", (media), lowBound=0, upBound=1, cat='Binary')

    # Objective Function
    prob += plp.lpSum(x[i]*(1/coef[i]) for i in media) # maximizing clicks

    # Constraints
    if spend_all==True:
        prob += plp.lpSum(x[i] for i in media) == Budget # Faz diferença ser obrigado ou não a gastar todo o budget
    else:
        prob += plp.lpSum(x[i] for i in media) <= Budget # Faz diferença ser obrigado ou não a gastar todo o budget

    for i in media:
        prob += (1-y[i])*bigM[i] + x[i] >= MinInvest[i]
        prob += (y[i]-1)*bigM[i] + x[i] <= MaxInvest[i]
        
        prob += y[i]*bigM[i] >= x[i]
        
    prob.solve()

    status = plp.LpStatus[prob.status]
    print(status)

    if status=='Optimal':
        df['Optimal Investment'] = df.Medias.apply(lambda j: x[j].value())
        df['Expected {}'.format(objective)] = round(df['Optimal Investment']/df[target[objective]],0)

        # st.write(
        #     '''
        #     ### Optimal Media Plan
        #     '''
        # )


        # write_title_h2('Optimal Media Plan')


        # st.table(df.style.hide_index())


        total_investiment = df['Optimal Investment'].sum()
        df['Share of Investment'] = df['Optimal Investment'] / total_investiment
        total_objective = df['Expected {}'.format(objective)].sum()
        df['Share of {}'.format(objective)] = df['Expected {}'.format(objective)] / total_objective
        # df['Expected {} (%)'.format(objective)] = df['Expected {} (%)'.format(objective)].round(decimals=3)


        cols = ['Medias', 'Optimal Investment', 'Share of Investment', 'Expected {}'.format(objective),
                'Share of {}'.format(objective)]

        # currency_cols = ['Optimal Investiment']
        # percent_cols = ['Optimal Investiment (%)', 'Expected {} (%)'.format(objective)]
        # formatted_df = df[cols]
        #
        # # currency_cols = ['CPM','CPC','CPA','MinInvestment','MaxInvestment']
        # for col_name in currency_cols:
        #     formatted_df = format_column_currency(df=formatted_df, col_name=col_name)
        #
        # for col_name in percent_cols:
        #     formatted_df = format_column_percent(df=formatted_df, col_name=col_name)
        #
        #
        # st.dataframe(formatted_df)

        with st.beta_expander("Optimal Media Plan", expanded=True):

            optimal = print_st_dataframe(
                df=df,
                index_col='Medias',
                cols=cols,
                currency_cols=['Optimal Investment'],
                percent_cols=['Share of Investment', 'Share of {}'.format(objective)],
                df_object_source=st)


        # print_currency_dataframe(df=df,
        #                          cols=cols,
        #                          currency_cols=['Optimal Investiment'],
        #                          percent_cols=['Optimal Investiment (%)', 'Expected {} (%)'.format(objective)],
        #                          object=st)


            if  status=='Optimal':

                cols = ['Medias', 'Optimal Investment', 'Share of Investment', 'Expected {}'.format(objective),
                        'Share of {}'.format(objective)]
                csv = df[cols].to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
                href = f'<a href="data:file/csv;base64,{b64}" download="optimal_investment.csv">Export optimal media plan as CSV</a> (right-click and save as &lt;some_name&gt;.csv)'
                st.markdown(href, unsafe_allow_html=True)
            elif  status=='Infeasible':
                st.error(
                    'Infeasible! Adjust budget and try again'
                )




        return x, status
    else:
        return x, status

@st.cache(allow_output_mutation=True)
def load_data(mydata):
    """
    docstring
    """
    df = pd.read_csv(mydata)
    default_bounds = df[['Medias','MinInvestment', 'MaxInvestment']].copy()

    return df, default_bounds

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Media Plann')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


def print_currency_dataframe(df, cols, currency_cols, percent_cols, object):
    # formatted_df = df[['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment']]
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


def download_mediaplan(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    # val = to_excel(df)
    b64 = base64.b64encode(df)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="extract.xlsx">Download csv file</a>' # decode b'abc' => abc


def print_st_dataframe(df, index_col, cols, currency_cols, percent_cols, df_object_source):


    formatted_df = df[cols]
    # formatted_df.set_index('Medias', inplace=True)
    for col_name in currency_cols:
        formatted_df = format_column_currency(df=formatted_df, col_name=col_name)

    for col_name in percent_cols:
        formatted_df = format_column_percent(df=formatted_df, col_name=col_name)

    # df_object = df_object_source.dataframe(formatted_df.set_index(index_col))

    df_object = df_object_source.table(formatted_df.set_index(index_col))
    # df_object = df_object_source.dataframe(formatted_df)

    local_css('style.css')

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
    st.markdown(href, unsafe_allow_html=True)
    placeholder.empty()


def write_title_h2(title):

    return st.markdown(''' 
    <div style="text-align:center; font-weight: bold">
    <h2 class='step'>'''+title+'''</h2> 
    </div> ''', unsafe_allow_html=True)


def format_column_currency(df, col_name):
    # df[col_name] = df[col_name].apply(lambda x: 'R$ ' + str(x).replace('.', ','))
    df[col_name] = df[col_name].apply(lambda x: str(x).replace('.', ','))
    df[col_name] = df[col_name].apply(lambda x:  str("{:,}".format(int(x.split(',')[0]))) + '|'  +  x.split(',')[1] )
    df[col_name] = df[col_name].apply(lambda x: x.replace(',','.').replace('|',',') )
    df[col_name] = df[col_name].apply(lambda x: 'R$ ' + str(x))
    return df

def format_column_percent(df, col_name):
    df[col_name] = df[col_name].apply(lambda x:  str((x*100)).replace('.', ','))
    df[col_name] = df[col_name].apply(lambda x: x.split(',')[0] + ',' + x.split(',')[1][0:1]   + '%')
    # df[col_name] = df[col_name].apply(lambda x:  str((x*100)).replace('.', ',') + '%')
    return df


def main():

    st.set_page_config(page_title='MIP - Media Investment Planner',
                       page_icon=':laptop:',
                       layout='wide',
                       initial_sidebar_state='collapsed')

    import streamlit_theme as stt

    stt.set_theme({'primary': '#6C7A96'})

    title = 'MIP - Media Investment Planner'
    navbar_html = '''
    <style>
        .navbar {
          overflow: hidden;
          background-color: #333;
          position: fixed; /* Set the navbar to fixed position */
          top: 0; /* Position the navbar at the top of the page */
          width: 100%; /* Full width */
        }

        /* Links inside the navbar */
        .navbar a {
          float: left;
          display: block;
          color: #f2f2f2;
          text-align: center;
          padding: 14px 16px;
          text-decoration: none;
        }

        /* Change background on mouse-over */
        .navbar a:hover {
          background: #ddd;
          color: black;
        }

        /* Main content */
        .main {
          margin-top: 30px; /* Add a top margin to avoid content overlay */
        }
    </style> 

    <div class="navbar">
        <a href="#">'''+title+'''</a>
    </div>

    '''

    # st.markdown(navbar_html, unsafe_allow_html=True)


    navbar_fixed = '''
    <div class="navbar">'''+title+'''</div>
    '''
    st.markdown(navbar_fixed, unsafe_allow_html=True)


    local_css('style.css')



    # step_1 = st.beta_expander("Step 1", expanded=True)
    # with step_1:
    #     col_file_1, col_file_2 = st.beta_columns([1,1])
    #     with col_file_1:
    #         data = st.file_uploader('Upload media file', type=['xlsx', 'csv'],key='data_temp')
    # if data is not None:
    #     df, default_bounds = load_data(data)
    #
    # else:
    #     download=st.sidebar.button('Download Excel Template File',key='Download_Button_Sidebar')
    #     with col_file_2:
    #         st.markdown('''<label class="css-145kmo2 effi0qh0">Or Download Template File</label>''', unsafe_allow_html=True)
    #         download=st.button('Download Excel Template File',key='Download_Button')
    #     if download:
    #         get_table_download_link()
    #
    #     return
    #
    # st.sidebar.markdown("### Step 2")
    # Budget = st.sidebar.number_input('Input budget limit:',value=default_bounds.MaxInvestment.sum())



    input_1_col, input_2_col, input_3_col, input_4_col, input_5_col = st.beta_columns([1,3,1,1,2])
    with input_1_col:
        step_1 = st.beta_expander("Step 1", expanded=True)
        with step_1:
            st.markdown('''<label class="css-145kmo2 effi0qh0">Template File</label>''',
                        unsafe_allow_html=True)

            placeholder = st.empty()
            get_table_download_link(placeholder)

            # download = placeholder.button('Download', key='Download_Button_new')
            # if download:
            #     get_table_download_link(placeholder)

    with input_2_col:
        step_2 = st.beta_expander("Step 2", expanded=True)
        with step_2:
            data = st.file_uploader('Upload media file', type=['xlsx', 'csv'], key='data')
            warnings = False
            if data is not None:
                df, default_bounds = load_data(data)


            # if data is not None:
            #     df, default_bounds = load_data(data)
            #
            # else:
            #     with col_file_2:
            #         st.markdown('''<label class="css-145kmo2 effi0qh0">Or Download Template File</label>''',
            #                     unsafe_allow_html=True)
            #         download = st.button('Download Excel Template File', key='Download_Button')
            #     if download:
            #         get_table_download_link()
            #
            #     return

    with input_3_col:
        # st.markdown("### Step 2")
        if data is not None:
            step_3 = st.beta_expander("Step 3", expanded=True)
            with step_3:

                Budget = st.number_input('Input budget limit:',value=default_bounds.MaxInvestment.sum(), key='budget')



    with input_4_col:
        if data is not None:
            step_4 = st.beta_expander("Step 4", expanded=True)
            with step_4:
                objective = st.selectbox('Campaign Objective:',
                                                 ['Clicks', 'Impressions', 'Acquisition']  # ,'Conversion']
                                                 ,key='objective')


    if data is not None:
        # spend_all = st.sidebar.checkbox('Spend entire budget?',value=True,key='budget_limit')
        spend_all = False

        target = {
        'Clicks':'CPC',
        'Impressions':'CPM',
        'Acquisition':'CPA',
        # 'Conversion':'Conversion rate',
        }

        # Budget = 10000.00
        media = df.Medias.values

        MinInvest = default_bounds[['Medias','MinInvestment']].set_index('Medias').to_dict()['MinInvestment']
        MaxInvest = default_bounds[['Medias','MaxInvestment']].set_index('Medias').to_dict()['MaxInvestment']


    with input_5_col:
        if data is not None:
            step_5 = st.beta_expander("Step 5", expanded=True)
            with step_5:
                spend_bounds = st.checkbox('Custom spending bounds', value=False, key='spending_bounds_check')


    if data is not None:

        if spend_bounds == False:

            for mchannel in media:
                index, = np.where(media == mchannel)
                df.at[index, 'MinInvestment'] = default_bounds['MinInvestment'][index]
                df.at[index, 'MaxInvestment'] = default_bounds['MaxInvestment'][index]


            with st.beta_expander('Planner', expanded=True):
                # mg_1, col_planner_no_bounds, mg_2 = st.beta_columns([1,25,1])
                # with col_planner_no_bounds:
                print_st_dataframe(
                    df=df,
                    index_col='Medias',
                    cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                    currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                    percent_cols=[],
                    df_object_source=st)



        else:

            # col1, col2 = st.beta_columns([3, 1])
            # col1_planner, col_low, col_upper, col_range = st.beta_columns([4, 1, 1, 1])
            col1_planner, col_low, col_upper = st.beta_columns([6, 1, 1])
            with col1_planner:

                # planner_2 = print_st_dataframe(
                #     df=df,
                #     index_col='Medias',
                #     cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                #     currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                #     percent_cols=[],
                #     df_object_source=st)
                with st.beta_expander('Planner',expanded=True):

                    # st.dataframe(df)
                    planner_2 = print_st_dataframe(
                        df=df,
                        index_col='Medias',
                        cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                        currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                        percent_cols=[],
                        df_object_source=st)

            with col_low:
                with st.beta_expander('Min Investment',expanded=True):

                    for mchannel in media:

                        index, = np.where(media == mchannel)
                        min = float(df['MinInvestment'][index[0]])
                        max = float(df['MaxInvestment'][index[0]])

                        low_number = st.number_input('{}'.format(mchannel),
                                                             min_value=0.0,
                                                             max_value=2.0 * MaxInvest[mchannel],
                                                             value=1.0 * min,
                                                             key='slider_min_{}'.format(mchannel))

                        # upper_number = st.number_input('{} Max Investiment'.format(mchannel),
                        #                                        min_value=0.0,
                        #                                        max_value=2.0 * MaxInvest[mchannel],
                        #                                        value=1.0 * MaxInvest[mchannel],
                        #                                        key='slider_max_{}'.format(mchannel))


                        if low_number >= max:
                            warnings = True
                            warning_low_greater_than_upper = True


                        index, = np.where(media == mchannel)
                        # df.at[index, 'MinInvestment'] =  low
                        # df.at[index, 'MaxInvestment'] =  upper
                        df.at[index, 'MinInvestment'] = low_number
                        # df.at[index, 'MaxInvestment'] = upper_number

                        planner_2 = print_st_dataframe(
                            df=df,
                            index_col='Medias',
                            cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                            currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                            percent_cols=[],
                            df_object_source=planner_2)


            with col_upper:
                with st.beta_expander('Max Investment',expanded=True):
                    for mchannel in media:

                        index, = np.where(media == mchannel)
                        min = float(df['MinInvestment'][index[0]])
                        max = float(df['MaxInvestment'][index[0]])

                        upper_number = st.number_input(label='',
                                                               min_value=0.0,
                                                               max_value=2.0 * MaxInvest[mchannel],
                                                               value=1.0 * max,
                                                               key='slider_max_{}'.format(mchannel))


                        if min >= upper_number:
                            warnings = True
                            warning_low_greater_than_upper = True

                        index, = np.where(media == mchannel)
                        df.at[index, 'MaxInvestment'] = upper_number


                        planner_2 = print_st_dataframe(
                            df=df,
                            index_col='Medias',
                            cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                            currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
                            percent_cols=[],
                            df_object_source=planner_2)


            # with col_range:
            #     with st.beta_expander('Investment Range',expanded=True):
            #         for mchannel in media:
            #             index, = np.where(media == mchannel)
            #             min = float(df['MinInvestment'][index[0]])
            #             max = float(df['MaxInvestment'][index[0]])
            #
            #
            #             low_number, upper_number = st.slider(label='',
            #                                             min_value=0.0, max_value=2.0*MaxInvest[mchannel],
            #                                             value=[1.0*min, 1.0*max],
            #                                             key='slider_{}'.format(mchannel)
            #                                             )
            #
            #
            #             index, = np.where(media == mchannel)
            #             df.at[index, 'MinInvestment'] = low_number
            #             df.at[index, 'MaxInvestment'] = upper_number
            #
            #
            #
            #             planner_2 = print_st_dataframe(
            #                 df=df,
            #                 index_col='Medias',
            #                 cols=['Medias', 'CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
            #                 currency_cols=['CPM', 'CPC', 'CPA', 'MinInvestment', 'MaxInvestment'],
            #                 percent_cols=[],
            #                 df_object_source=planner_2)



        col1_margin,col2_btn,col3_margin = st.beta_columns([1,1,1])
        with col2_btn:
            optimize_button = st.button("Run Media Optimization",key='optimize_button')




        if Budget <= 0:
            warnings = True

        # if low_number >= upper_number:
        #     warnings = True

        if optimize_button and warnings == True:
            if Budget <= 0:
                st.warning("Input Budget must be greater than zero")
            if warning_low_greater_than_upper:
                st.warning("Min Investment must be lower than Max Investment")
        elif optimize_button and warnings == False:
            x, status = run_optimization(media, target, objective, Budget, spend_all, df)




    
        return

if __name__ == "__main__":

    main()
        

