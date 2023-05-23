import streamlit as st
import pandas as pd
import numpy as np    
import matplotlib.pyplot as plt
import altair as alt
from PIL import Image
from Front_end.front_end_functions import df_check_data
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1- Main Window -- Layout Settings------------------------------------------------------------
st.set_page_config(layout="wide")
base="dark"
primaryColor="#BF2A7C" #PINK
backgroundColor="#FFFFFF" #MAIN WINDOW BACKGROUND COLOR (white)
secondaryBackgroundColor="#EBF3FC" #SIDEBAR COLOR (light blue)
textColor="#31333F"
secondaryColor="#F0F2F6" #dark_blue
tertiaryColor ="#0810A6"
light_pink = "#CDC9FA"
plot_blue_colour="#0810A6" #vibrant blue for plots

footer="""<style>.footer {
position: fixed;left: 0;bottom: 0;width: 100%;background-color: white;color: black;text-align: center;
}
</style>
<div class="footer">
<p>(c) 2023 Zeta Global, Dev Version 1.1, GDSA</p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)


# Input Data 
st.sidebar.title("Input Data")


## Unmatched file: this are unknown customers 
unmatched_file = st.sidebar.file_uploader('Unmatched Dataset', type='csv', help='Dataset without email address')

unmatched_file_valid_flag = False
if unmatched_file is not None:
    # Check MIME type of the uploaded file
    if  unmatched_file.name == "unmatched_data.csv":
        unmatched_df = pd.read_csv(unmatched_file)
        unmatched_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called unmatched_data.csv')

## Customer file: this are known customers (a.k.a customer databse with PII)
customer_file = st.sidebar.file_uploader("Customer Dataset", type='csv', help='Dataset containing email address')

customer_file_valid_flag = False
if customer_file is not None:
    # Check MIME type of the uploaded file
    if  customer_file.name == "customer_data.csv":
        customer_df = pd.read_csv(customer_file)
        customer_file_valid_flag = True
        st.sidebar.success('Success')
    else:
        st.sidebar.error('Error: please, re-upload file called customer_data.csv')



## Check Data Formats button
check_data = st.sidebar.button("""Check Data""", help = 'Show statistics of your data')


# Persist  unm/cust data statistics
if 'valid_flag' not in st.session_state:
    st.session_state.valid_flag = False

if 'unmatched_obj_cols' not in st.session_state:
    st.session_state.unmatched_obj_cols = ['']
if 'unmatched_num_cols' not in st.session_state:
    st.session_state.unmatched_num_cols = ['']
if 'unmatched_obj_cols_optional' not in st.session_state:
    st.session_state.unmatched_obj_cols_optional = ['']

if 'unmatched_df_com_cols' not in st.session_state:
    st.session_state.unmatched_df_com_cols = pd.DataFrame()
if 'customer_df_com_cols' not in st.session_state:
    st.session_state.customer_df_com_cols = pd.DataFrame()

if check_data:
    if (unmatched_file_valid_flag == True) and (customer_file_valid_flag ==True):

        unmatched_df, customer_df, unmatched_obj_cols, unmatched_num_cols, unmatched_obj_cols_optional, commun_cols_u_c= df_check_data(unmatched_df, customer_df)
    
        st.session_state.valid_flag = True
        st.session_state.unmatched_obj_cols = unmatched_obj_cols
        st.session_state.unmatched_num_cols = unmatched_num_cols
        st.session_state.unmatched_obj_cols_optional = unmatched_obj_cols_optional
        st.session_state.unmatched_df_com_cols = unmatched_df[commun_cols_u_c]
        st.session_state.customer_df_com_cols = customer_df[commun_cols_u_c]

    else:
        pass
else:
    pass

# -----------------------------------------------------------------------------------------------
# 1- Main Window -- Parameter Settings-----------------------------------------------------------
# ----------------------------------------------------------------------------------------------- 
# st.title(f'My first app {st.__version__}')


# Creating columns 1 and 2
col1, col2 = st.columns([13, 2])

## Zeta Logo
#zeta_logo = Image.open('ZETA_BIG-99e027c9.webp') #white logo 
zeta_logo = Image.open('ZETA_BIG-99e027c92.png') #blue logo 
col2.image(zeta_logo)

## Header
col1.title("Zeta Customer Matcher")
"""This app demonstrates Customers Probabilistic Matching Project"""

# Creating columns 3 and 4
col_space1, col_space2  =  st.columns([2,2])

# Creating columns 3 and 4
col3, col4= st.columns([2, 2])
colna1, colna2 = st.columns([2,2]) # NA message if no NAs

# Create col expand data
col_left_expander, col_right_expander = st.columns(2)

# Creating columns 5 and 6

col_threshold_left, col_threshold_rigt = st.columns([2,2])

# Creating columns 5 and 6

col5, col6 = st.columns([2,2])


## Summary pie charts
colp1, colp2, colp3 = st.columns([2, 2, 2])


# Set num of column algorithm will compare, by default is 1
col_compare_alg = 1


if st.session_state['valid_flag']:
    #If both files are uploaded print stats of each one
    if  unmatched_file_valid_flag and customer_file_valid_flag:
        col_space1.success('Format Check Completed')    
        df_info = {'Number of rows':[len(st.session_state['unmatched_df_com_cols']), len(st.session_state.customer_df_com_cols)],
                    'Number of columns': [len(st.session_state['unmatched_df_com_cols'].columns), len(st.session_state.customer_df_com_cols.columns)]}
        df_info = pd.DataFrame(df_info).transpose().rename(columns={0:'Unmatch Data', 1:'Customer Data'})
        col_space1.dataframe(df_info)

        ## NA charts
        null_df_unmatched = st.session_state['unmatched_df_com_cols'].apply(lambda x: sum(x.isnull())).to_frame(name='count_null').reset_index()
        null_df_customers = st.session_state.customer_df_com_cols.apply(lambda x: sum(x.isnull())).to_frame(name='count_null').reset_index()

        #null_counts = df.isnull().sum()
        notnull_df_unmatched = st.session_state['unmatched_df_com_cols'].apply(lambda x: sum(x.notnull())).to_frame(name='count').reset_index()
        notnull_df_customers = st.session_state.customer_df_com_cols.apply(lambda x: sum(x.notnull())).to_frame(name='count').reset_index()

        # create new DataFrame with counts
        #counts_df = pd.DataFrame({'null_counts': null_counts,
        #                  'notnull_counts': notnull_counts})

        merged_unmatched = pd.merge(null_df_unmatched, notnull_df_unmatched, on='index')
        merged_customer = pd.merge(null_df_customers, notnull_df_customers, on='index')
        merged_unmatched1=merged_unmatched.melt('index',var_name='type',value_name='count')
        merged_customer1=merged_customer.melt('index',var_name='type',value_name='count')

        unmatched_city_count = st.session_state.unmatched_df_com_cols.groupby(['STORE_CITY'])['STORE_CITY'].count().sort_values(ascending=False).reset_index(name='counts').head(10)
        customer_city_count = st.session_state.customer_df_com_cols.groupby(['STORE_CITY'])['STORE_CITY'].count().sort_values(ascending=False).reset_index(name='counts')#.head(10)
        # Merge DataFrame A and DataFrame B on 'city_store' column
        merged_df = pd.merge(unmatched_city_count, customer_city_count, on='STORE_CITY')
        # Create final DataFrame B with selected columns
        final_customer_city_count = merged_df[['STORE_CITY', 'counts_y']].rename(columns={'counts_y': 'counts'})


        unmatched_city_avg_buys = st.session_state.unmatched_df_com_cols.groupby(['STORE_CITY'])['NUM_SHOP'].mean().round(0).sort_values(ascending=False).reset_index(name='avg_num_shop')#.head(10)
        unmatched_store_city_count = st.session_state.unmatched_df_com_cols.groupby(['STORE_CITY', 'STORE'])['STORE'].count().sort_values(ascending=False).reset_index(name='counts')
        unmatched_num_shop_count = st.session_state.unmatched_df_com_cols.groupby(['NUM_SHOP'])['NUM_SHOP'].count().sort_values(ascending=False).reset_index(name='counts')#.head(10)
        customer_num_shop_count = st.session_state.customer_df_com_cols.groupby(['NUM_SHOP'])['NUM_SHOP'].count().sort_values(ascending=False).reset_index(name='counts')#.head(10)

        merged_df_un = pd.merge(st.session_state.unmatched_df_com_cols, unmatched_city_count, on='STORE_CITY', how='inner')
        # Select desired columns from df1
        #merged_df_un = merged_df[st.session_state.unmatched_df_com_cols.columns]
        #merged_df_un

        merged_df_cu = pd.merge(st.session_state.customer_df_com_cols, final_customer_city_count, on='STORE_CITY', how='inner')
        # Select desired columns from df1
        #merged_df_cu = merged_df[st.session_state.customer_df_com_cols.columns]


        #merged_unmatched['count'].idxmax()
        #merged_customer['count'].idxmax()
        cmap = {
        'count_null': '#BF2A7C',
        'count': '#0810A6',
        }
        #altair horizontal bar chart nulls and not nulls pink and blue 
        na_unmatched=alt.Chart(merged_unmatched1).mark_bar(size=20).encode(
            x=alt.X('count:Q', scale= alt.Scale(domainMax= len(st.session_state['unmatched_df_com_cols']) )),
            y='type:N',  
            color=alt.Color('type:N', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
            tooltip=['count'] ,
            row=alt.Row('index:N', title=None)
        ).properties(width=500 , height=alt.Step(25) ).configure_axis(title=None, grid=False)

        #plotly gauge charts with number of not nulls
        fig = make_subplots(rows=2, cols=3, specs = [[{"type": "indicator"}] * 3] * 2 ) #specs=[[{"type": "indicator"} for c in df.columns] for t in df.index]

        for r, row in notnull_df_unmatched.iterrows():
                variable = row['index']
                value = row['count']
    
                fig.add_trace(
                    go.Indicator(mode="gauge+number", value=value, gauge = {'axis': {'range': [None, len(st.session_state['unmatched_df_com_cols'])]},'bar': {'color': "#0810A6"},}, title={"text": f"{variable}"}), row=r // 3 + 1, col=r % 3 + 1 )

        fig.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20} , height=500, width=900, title_text="Unmatched Charts")

        #Radial chart with Store_city and its count
        base = alt.Chart(unmatched_city_count).encode(
        theta=alt.Theta("counts:Q", stack=True ),
        radius=alt.Radius("counts", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color=alt.Color ("STORE_CITY:N"),
        order= alt.Order("counts:Q", sort="ascending")
        )
        c1 = base.mark_arc(innerRadius=20, stroke="#fff")
        c2 = base.mark_text(radiusOffset=10).encode(text="counts:Q")

        #Radial chart with Store_city and average number of transaction per city
        base_avg = alt.Chart(unmatched_city_avg_buys).encode(
        theta=alt.Theta("avg_num_shop:Q", stack=True ),
        radius=alt.Radius("avg_num_shop", scale=alt.Scale(type="sqrt", zero=True, rangeMin=10)),
        color=alt.Color ("STORE_CITY:N", scale=alt.Scale(scheme='plasma')),
        order= alt.Order("avg_num_shop:Q", sort="ascending")
        )
        c1_avg = base_avg.mark_arc(innerRadius=20, stroke="#fff")
        c2_avg = base_avg.mark_text(radiusOffset=10).encode(text="avg_num_shop:Q")
        
        #portion of the population for each NUM_SHOP value
        base_num = alt.Chart(unmatched_num_shop_count).encode(
        theta=alt.Theta("counts:Q", stack=True ),
        radius=alt.Radius("counts", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color=alt.Color ("NUM_SHOP:N"), #
        order= alt.Order("counts:Q", sort="ascending")
        )
        c1_num = base_num.mark_arc(innerRadius=20, stroke="#fff")
        c2_num = base_num.mark_text(radiusOffset=10).encode(text="counts:Q")

        
        
        #Stack bar with city, Numshop
        bars = alt.Chart(merged_df_un.dropna()).mark_bar().encode(
        x=alt.X('count()', stack='zero'),
        y=alt.Y('STORE_CITY:N'),
        color=alt.Color('NUM_SHOP:N', bin=alt.Bin(maxbins=5), scale=alt.Scale(scheme='plasma'))
            )


        #na_unmatched = na_graphs(null_df_unmatched, name='unmatched')
        with col3:
            #st.altair_chart(na_unmatched, use_container_width=False)
            #st.plotly_chart(fig, use_container_width=True)
            #unmatched_city_count
            st.altair_chart((c1+c2), use_container_width=True)
            #unmatched_city_avg_buys
            #st.altair_chart((c1_avg+c2_avg), use_container_width=True)
            #unmatched_num_shop_count
            #st.altair_chart((c1_num+c2_num), use_container_width=True)
            st.altair_chart((bars), use_container_width=True)
            #null_df_unmatcheds
            #notnull_df_unmatched
            #merged_unmatched
            #merged_unmatched1
            st.write('')

        #altair horizontal bar chart nulls and not nulls pink and blue 
        na_customer=alt.Chart(merged_customer1).mark_bar(size=20).encode(
            x=alt.X('count:Q', scale= alt.Scale(domainMax= len(st.session_state.customer_df_com_cols) )),
            y='type:N',  
            color=alt.Color('type:N', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
            tooltip=['count'] ,
            row=alt.Row('index:N', title=None)
        ).properties(width=500, height=alt.Step(25)).configure_axis(title=None, grid=False)

        #plotly gauge charts with number of not nulls
        fig2 = make_subplots(rows=2, cols=3, specs = [[{"type": "indicator"}] * 3] * 2 ) #specs=[[{"type": "indicator"} for c in df.columns] for t in df.index]

        for r, row in notnull_df_customers.iterrows():
                variable = row['index']
                value = row['count']
    
                fig2.add_trace(
                    go.Indicator(mode="gauge+number", value=value, gauge = {'axis': {'range': [None, len(st.session_state['customer_df_com_cols'])]},'bar': {'color': "#0810A6"},}, title={"text": f"{variable}"}), row=r // 3 + 1, col=r % 3 + 1 )

        fig2.update_layout(margin={"l": 20, "r": 20, "t": 40, "b": 20} , height=500, width=900, title_text="Customer Charts")

        #Radial chart with Store_city and its count
        base_c = alt.Chart(final_customer_city_count).encode(
        theta=alt.Theta("counts:Q", stack=True ),
        radius=alt.Radius("counts", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color=alt.Color ("STORE_CITY:N"),
        order= alt.Order("counts:Q", sort="ascending")
        )
        c1_c = base_c.mark_arc(innerRadius=20, stroke="#fff")
        c2_c = base_c.mark_text(radiusOffset=10).encode(text="counts:Q")

        #portion of the population for each NUM_SHOP value
        base_num_c = alt.Chart(customer_num_shop_count).encode(
        theta=alt.Theta("counts:Q", stack=True ),
        radius=alt.Radius("counts", scale=alt.Scale(type="sqrt", zero=True, rangeMin=20)),
        color=alt.Color ("NUM_SHOP:N"), #
        order= alt.Order("counts:Q", sort="ascending")
        )
        c1_num_c = base_num_c.mark_arc(innerRadius=20, stroke="#fff")
        c2_num_c = base_num_c.mark_text(radiusOffset=10).encode(text="counts:Q")

        #Stack bar with city, Numshop
        bars_2 = alt.Chart(merged_df_cu.dropna()).mark_bar().encode(
        x=alt.X('count()', stack='zero'),
        y=alt.Y('STORE_CITY:N'),
        color=alt.Color('NUM_SHOP:N', bin=alt.Bin(maxbins=5), scale=alt.Scale(scheme='plasma'))
            )

        with col4:
            #st.altair_chart(na_customer, use_container_width=False)
            #st.plotly_chart(fig2, use_container_width=True)
            #null_df_customers
            #notnull_df_customers
            #merged_customer
            #merged_customer1
            #final_customer_city_count
            st.altair_chart((c1_c+c2_c), use_container_width=True)
            st.altair_chart((bars_2), use_container_width=True)
            #st.altair_chart((c1_num_c+c2_num_c), use_container_width=True)
            st.write('')
        
        #merged_customer_df  = merged_customer.assign(df=np.full(len(merged_customer), 'Customer'))
        #merged_unmatched_df  = merged_unmatched.assign(df=np.full(len(merged_unmatched), 'Unmatched'))

        #result = pd.concat([merged_customer_df, merged_unmatched_df]).reset_index()
        
        

        #na_chart = alt.Chart(result).mark_bar().encode(
        #x='sum(count):Q',
        #y='df:N',
        #color=alt.Color('type', scale=alt.Scale(domain=list(cmap.keys()), range=list(cmap.values()))) ,
        #row='index:N'
        #).configure_axis(title=None, grid=False).properties(width=700, height=70 )

        #st.altair_chart(na_chart)
        #st.write('')

        # Unmatch df
        col_left_expander.write('Unmatched data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            st.dataframe(st.session_state['unmatched_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Unmatched Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['unmatched_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)
        
        if len(na_unmt_df[na_unmt_df['#NAs']>0]) == 0:
            colna1.info('No NAs found in Unmatched Dataset')

        # Customer df
        col_right_expander.write(f'Customer data')
        with col_right_expander.expander("Expand data and statistics"):
            #Display customer df
            st.dataframe(st.session_state.customer_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Customer Data: ')
            na_cust_df = pd.DataFrame(st.session_state.customer_df_com_cols.isna().sum())
            na_cust_df = na_cust_df.rename(columns={0:'#NAs'})
            st.dataframe(na_cust_df)
        if len(na_cust_df[na_cust_df['#NAs']>0]) == 0:
            colna2.info('No NAs found in Customer Dataset')


    # Print unmatched_file stats if uploaded
    elif unmatched_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_left_expander.write('Unmatched data')
        with col_left_expander.expander("Expand data and statistics"):
            #Display unmatch df
            st.dataframe(st.session_state['unmatched_df_com_cols'].head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Unmatched Dataset: ')
            na_unmt_df = pd.DataFrame(st.session_state['unmatched_df_com_cols'].isna().sum())
            na_unmt_df = na_unmt_df.rename(columns={0:'#NAs'})
            st.dataframe(na_unmt_df)

    # Print customer_file stats if uploaded
    elif customer_file_valid_flag:
        col_space1.success('Format Check Completed')
        col_right_expander.write(f'Customer data')
        with col_right_expander.expander("Expand data and statistics"):
            #Display customer df
            st.dataframe(st.session_state.customer_df_com_cols.head(100))
            #Display unmatch df NAs 
            st.write(f'Number of Nan values in Customer Data: ')
            na_cust_df = pd.DataFrame(st.session_state.customer_df_com_cols.isna().sum())
            na_cust_df = na_cust_df.rename(columns={0:'#NAs'})
            st.dataframe(na_cust_df)

    else: 
         pass
else:
    col3.write('')


