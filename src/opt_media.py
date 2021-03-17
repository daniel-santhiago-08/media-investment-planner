import pulp as plp
import numpy as np

def run_optimization(params):

    media = params['state'].media   
    target = params['target']  
    objective = params['objective']   
    Budget = params['Budget']  
    spend_all = params['spend_all']   
    df = params['state'].df 

    spend_all = False

    objectives = list(target.keys())
    objectives.append('Multi-Objective')

    output={obj:None for obj in objectives}
    # print(objectives)
    # print(output)

    for obj in objectives:

        if obj!='Multi-Objective':
            coef = df[['Medias',target[obj]]].set_index('Medias').to_dict()[target[obj]]
        else:
            normalized_df = df[list(target.values())].copy()
            # normalized_df=np.log1p(normalized_df/normalized_df.max())
            normalized_df=normalized_df/normalized_df.max()

            coef = df[['Medias']+list(target.values())].copy()
            
            for tg in list(target.values()):
                coef[tg] = normalized_df[tg]

            coef = coef.set_index('Medias')

            coef = 1/coef
            coef = coef.sum(axis=1).to_dict()

        # print(coef)
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

        output[obj] = (x,status)
        # import streamlit as st
        # st.write(obj)
        # st.write(output[obj][0])
        # st.write(output[obj][0].keys())
        # k = output[obj][0].keys()
        # for ki in k:
        #     st.write(output[obj][0][ki].value())


    return output