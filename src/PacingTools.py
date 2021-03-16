import pandas as pd
import streamlit as st
import math

def getPacing(method, budget, medias, share_medias, duration):
    return globals()[method](budget, medias, share_medias, duration)

def Uniform(budget, medias, share_medias, duration):

    pacing_list = []
    for day in range(1 ,duration +1):
        pacing_dict = {}
        pacing_dict['day'] = day
        for media in medias:
            pacing_dict[media] = (share_medias[media][0] * budget) / duration
        pacing_list.append(pacing_dict)
    df_pacing = pd.DataFrame(pacing_list)
    return df_pacing

def Increasing(budget, medias, share_medias, duration):

    pacing_list = []
    for day in range(1 ,duration +1):
        pacing_dict = {}
        pacing_dict['day'] = day
        for media in medias:
            # pacing_dict[media] = (share_medias[media][0] * budget) / duration
            pacing_dict[media] = day * (share_medias[media][0] * budget * 2) / ((duration + 1) * duration)

        pacing_list.append(pacing_dict)
    df_pacing = pd.DataFrame(pacing_list)
    return df_pacing

def Decreasing(budget, medias, share_medias, duration):

    pacing_list = []
    for day in range(1 ,duration +1):
        pacing_dict = {}
        pacing_dict['day'] = day
        for media in medias:
            # pacing_dict[media] = (share_medias[media][0] * budget) / duration
            pacing_dict[media] = (duration - day + 1) * (share_medias[media][0] * budget * 2) / ((duration + 1) * duration)

        pacing_list.append(pacing_dict)
    df_pacing = pd.DataFrame(pacing_list)
    return df_pacing


def V_Shaped(budget, medias, share_medias, duration):

    pacing_list = []
    for day in range(1 ,duration +1):
        pacing_dict = {}
        pacing_dict['day'] = day
        for media in medias:
            # pacing_dict[media] = (share_medias[media][0] * budget) / duration

            if (duration%2) == 0:
                even = True
                odd = False
            else:
                even = False
                odd = True

            if odd:
                middle = math.ceil(duration/2)
                # middle = math.floor(duration/2)

                if day <= middle:
                    # Decreasing
                    pacing_dict[media] = ((middle - day + 1) ) * ( (share_medias[media][0] * budget )  / ((2+ middle ) * (middle - 1) + 1)  )

                elif day > middle:
                    # Increasing
                    pacing_dict[media] = (( day - (duration - middle))  )  * ((share_medias[media][0] * budget )  / ((2 + middle ) * (middle - 1) + 1) )

            if even:
                middle = math.ceil(duration/2)
                # middle = math.floor(duration/2)

                if day <= middle:
                    # Decreasing
                    pacing_dict[media] = ((middle - day + 1) ) * ( (share_medias[media][0] * budget) / ((middle)*(middle+1)) )

                elif day > middle:
                    # Increasing
                    pacing_dict[media] = (( day - (duration - middle))  )  * ((share_medias[media][0] * budget) / ((middle)*(middle+1)) )

        pacing_list.append(pacing_dict)
    df_pacing = pd.DataFrame(pacing_list)
    return df_pacing


def Inverted_V_Shaped(budget, medias, share_medias, duration):

    pacing_list = []
    for day in range(1 ,duration +1):
        pacing_dict = {}
        pacing_dict['day'] = day
        for media in medias:
            # pacing_dict[media] = (share_medias[media][0] * budget) / duration

            if (duration%2) == 0:
                even = True
                odd = False
            else:
                even = False
                odd = True

            if odd:
                middle = math.ceil(duration/2)
                # middle = math.floor(duration/2)

                if day <= middle:
                    # Increasing
                    # pacing_dict[media] = ((day - (duration - middle))) * ((share_medias[media][0] * budget) / ( ((middle) * (middle - 1)) + middle))
                    pacing_dict[media] = ( day ) * ((share_medias[media][0] * budget) / ( ((middle) * (middle - 1)) + middle))


                elif day > middle:
                    # Decreasing
                    pacing_dict[media] = (duration - day + 1) * ((share_medias[media][0] * budget) / (((middle)*(middle - 1)) + middle))


            if even:
                middle = math.ceil(duration / 2)
                # middle = math.floor(duration/2)

                if day <= middle:
                    # Increasing
                    # pacing_dict[media] = ((day - (duration - middle))) * ((share_medias[media][0] * budget) / ( ((middle) * (middle - 1)) + middle))
                    pacing_dict[media] = (day) * ((share_medias[media][0] * budget) / (middle * (middle+1)))


                elif day > middle:
                    # Decreasing
                    pacing_dict[media] = (duration - day + 1) * ((share_medias[media][0] * budget) / (middle * (middle+1)))


        pacing_list.append(pacing_dict)
    df_pacing = pd.DataFrame(pacing_list)
    return df_pacing

def pacing_calculation(params):

    pacing_dict = {}
    objectives = params['state'].optimization.keys()
    # st.write(params['state'].optimization)

    for objective in objectives:

        df = params['state'].optimization[objective]

        budget = df['Optimal Investment'].sum()
        medias = df['Medias'].tolist()
        share_medias = df[['Medias','Share of Investment']].set_index('Medias').T.to_dict('list')
        duration = params['duration']
        duration = int(duration)
        pacing_type = params['pacing']

        
        df_pacing = getPacing(pacing_type, budget, medias, share_medias, duration)

        params['df_pacing'] = df_pacing

        pacing_dict[objective] = df_pacing

    params['state'].pacing_dict = pacing_dict

    return params
