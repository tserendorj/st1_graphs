import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2 
from io import BytesIO 

primaryColor="#BF2A7C" #PINK
backgroundColor="#FFFFFF" #MAIN WINDOW BACKGROUND COLOR (white)
secondaryBackgroundColor="#EBF3FC" #SIDEBAR COLOR (light blue)
textColor="#31333F"
secondaryColor="#F0F2F6" #dark_blue
tertiaryColor ="#0810A6"
light_pink = "#CDC9FA"
plot_blue_colour="#0810A6" #vibrant blue for plots


def df_check_data(unmatched_df, customer_df):
    unmatched_df.columns = map(str.upper, unmatched_df.columns)
    customer_df.columns = map(str.upper, customer_df.columns)  
    commun_cols_u_c = list(unmatched_df.columns[unmatched_df.columns.isin(customer_df.columns)])
    unmatched_df = unmatched_df[commun_cols_u_c]
    customer_df = customer_df[commun_cols_u_c]
    unmatched_obj_cols = list(unmatched_df.select_dtypes(include="object").columns)
    unmatched_num_cols = list(unmatched_df.select_dtypes(exclude="object").columns)
    customer_obj_cols = list(customer_df.select_dtypes(include="object").columns) 
    customer_num_cols = list(customer_df.select_dtypes(exclude="object").columns)
    assert (unmatched_obj_cols == customer_obj_cols),"String column data types do not match"
    assert (unmatched_num_cols == customer_num_cols),"Numeric column data types do not match"

    unmatched_num_cols.insert(0,"")
    unmatched_obj_cols_optional = unmatched_obj_cols.copy()
    unmatched_obj_cols_optional.insert(0,"")

    return unmatched_df, customer_df, unmatched_obj_cols, unmatched_num_cols, unmatched_obj_cols_optional, commun_cols_u_c


def comp_time_eq(n_rows, threshold):
        if str(threshold) == '0.95':
            z_95 = np.array([ 8.61430182e-04, -1.21567894e+02])
            res = z_95[0] * n_rows + z_95[1]
            if res >= 0:
                return res
            else:
                return 0
        elif str(threshold == '0.75') or str(threshold) == '0.85':
            z_75 = np.array([ 6.15975524e-04, -2.13769878e+01])
            res = z_75[0] * n_rows + z_75[1]
            if res >= 0:
                return res
            else:
                return 0
        elif (str(threshold) == '0.50' or str(threshold) == '0.60') and n_rows < 3000000:
            z_50_a3M = np.array([ 7.43574104e-04, -4.45166486e+01])
            res =  z_50_a3M[0] * n_rows + z_50_a3M[1]
            if res >= 0:
                return res
            else:
                return 0
        elif (str(threshold) == '0.50' or str(threshold) == '0.60') and n_rows >= 3000000:
            z_50_d3M = np.array([ 5.31421923e-03, -1.36647504e+04])
            res = z_50_d3M[0] * n_rows + z_50_d3M[1]
            if res >= 0:
                return res
            else:
                return 0
            


def threshold_match_tbl(threshold_possibilities,count_num_mtch ):
    match_dict = dict(zip(threshold_possibilities, count_num_mtch))
    data = {'Threshold': match_dict.keys(), '#Match': match_dict.values()}
    best_threshold_df = pd.DataFrame.from_dict(data, orient='index')
    best_threshold_df.columns = [''] * len(best_threshold_df.columns)
    best_threshold_df_T = best_threshold_df.copy() #Save here best_threshold_df transpose to use it later for a graph
    new_header = best_threshold_df.iloc[0] #grab the first row for the header
    best_threshold_df = best_threshold_df[1:] #take the data less the header row
    best_threshold_df.columns = new_header
    best_threshold_df = best_threshold_df.reset_index().rename(columns={'index':'Threshold'})

    return match_dict, best_threshold_df, best_threshold_df_T


def df_store_match (df_final,select_box_unmatched_load_main):
    df_copy = df_final.copy()
    df_copy['100_match'] = np.where (df_final[f'{select_box_unmatched_load_main}_MTCH'] == df_final[f'{select_box_unmatched_load_main}_UNMTCH'], 1, 0)
    df_copy['100_match_address'] = np.where(df_copy.STORE_CITY_MTCH == df_copy.STORE_CITY_UNMTCH, 1, 0)

    df_gb_store_mtch = pd.DataFrame()
    # Group by Store --- Name matching
    df_gb_store_mtch['TOTAL_CUSTOMERS'] = df_copy.groupby('STORE_MTCH')['STORE_MTCH'].count()
    df_gb_store_mtch['100%_MATCH'] =df_copy.groupby('STORE_MTCH')['100_match'].sum()
    df_gb_store_mtch['MISSPELLING'] =  df_gb_store_mtch['TOTAL_CUSTOMERS']  - df_gb_store_mtch['100%_MATCH']
    df_gb_store_mtch['MISSPELLING_%'] =  np.round(df_gb_store_mtch['MISSPELLING'] * 100 / df_gb_store_mtch['TOTAL_CUSTOMERS'], 2)

    df_gb_store_mtch = df_gb_store_mtch.sort_values('MISSPELLING_%', ascending=False)
    df_gb_store_mtch = df_gb_store_mtch.reset_index().rename(columns={'STORE_MTCH': 'STORE'})

    return df_copy, df_gb_store_mtch


def venn_diagram (unmatched_df_count,matched_df_count,match_count ):
    fig, ax = plt.subplots()
    fig.patch.set_facecolor(backgroundColor)

    plt.title('Matching', color = textColor, fontsize=10)
    plt.figure(figsize=(10,10))

    total= 100.0
    v = venn2(subsets=(unmatched_df_count, matched_df_count, match_count), 
                    set_labels=("Unmatched Records", "Customer Database"), 
                    # subset_label_formatter=lambda x: f"{(np.round(x/1000000, 1))}M", #uncomment if you want to round you len(df) to Millions
                    subset_label_formatter=lambda x: f"{(np.round(x/1000, 1))}K",
                    ax=ax,set_colors=(primaryColor, plot_blue_colour), 
                    alpha = 0.99
                    )
    i = 0
    for text in v.set_labels:
            text.set_color(textColor)
            text.set_fontsize(9) 
            i+=1

    for text in v.subset_labels:   
            text.set_color('white')   
            text.set_fontsize(9)   
            # text.set_fontweight('bold')
    plt.show()

    return fig


def mini_pie_charts(df_final, select_box_unmatched_load_main, unmtch_name_col, ctomer_name_col ):
     
    df_final['cmplete_name_match'] = np.where(df_final[f'{select_box_unmatched_load_main}_MTCH'] == df_final[f'{select_box_unmatched_load_main}_UNMTCH'], 1, 0)

    cmplete_name_match= len(df_final[df_final['cmplete_name_match']==1])
    mtch_output_len = len(df_final)
    cmplete_name_match_percentage = np.round(cmplete_name_match*100/mtch_output_len, 2)

        # pie chart 1
    y = np.array([unmtch_name_col.nunique(), len(unmtch_name_col) - unmtch_name_col.nunique()])
    labels = ['Unmatched unique customers', 'Duplicates']
    mycolors = [plot_blue_colour, primaryColor]
    explode = (0, 0)
    
        #fig1, ax1 = plt.subplots()
    fig1, ax1 = plt.subplots(figsize=(6,6))
    fig1.patch.set_facecolor(backgroundColor)
    
    ax1.pie(y, autopct=lambda x: '{:.0f}'.format(x*y.sum()/100), explode=explode, labels=None,
            shadow=False, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14} ) 
    ax1.axis('equal')
    plt.title('Unmatched dataset', color=textColor, y=1.1, fontsize=16)
    plt.legend(labels, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))

    buf1 = BytesIO()
    fig1.savefig(buf1, format="png")
       

        #colp1.pyplot(fig1)
    
        # pie chart 2
    y = np.array([ctomer_name_col.nunique(), len(ctomer_name_col) - ctomer_name_col.nunique()])
    labels = ['Known unique customers', 'Duplicates']
    mycolors = [plot_blue_colour, primaryColor]
    explode = (0, 0)
    
        #fig2, ax2 = plt.subplots()
    fig2, ax2 = plt.subplots(figsize=(6,6))
    fig2.patch.set_facecolor(backgroundColor)
    
    ax2.pie(y, autopct=lambda x: '{:.0f}'.format(x*y.sum()/100), explode=explode, labels=None,
            shadow=False, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14}, radius=1800)
    ax2.axis('equal')
    plt.title('Customer dataset', color=textColor, y=1.1, fontsize=16)
    plt.legend(labels, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))
    
    buf2 = BytesIO()
    fig2.savefig(buf2, format="png")
    

        #colp2.pyplot(fig2)
    
        # pie chart 3
    y = np.array([len(df_final['cmplete_name_match']) - cmplete_name_match, cmplete_name_match])
    labels = ['Not exact match', '100% exact match']
    mycolors = [plot_blue_colour, primaryColor]
    explode = (0, 0)
    
        #fig3, ax3 = plt.subplots()
    fig3, ax3 = plt.subplots(figsize=(6,6))
    fig3.patch.set_facecolor(backgroundColor)
    
    
    ax3.pie(y, autopct=lambda x: '{:.0f}'.format(x*y.sum()/100), explode=explode, labels=None,
            shadow=False, startangle=90, colors=mycolors, textprops={'color':"w",'fontsize': 14},  radius=1800)
    ax3.axis('equal')
    plt.title('Threshold dataset', color=textColor, y=1.1, fontsize=16)
    plt.legend(labels, loc='lower left', bbox_to_anchor=(0.5, -0.08))
        #plt.figure(figsize=(10,10))

    buf3 = BytesIO()
    fig3.savefig(buf3, format="png")

    return buf1, buf2, buf3


def violin_graph (df_gb_store_mtch):
    violin_plot = df_gb_store_mtch[['TOTAL_CUSTOMERS', 'MISSPELLING']]
    ax_violin_plot = plt.figure()
    ax_violin_plot.patch.set_facecolor(backgroundColor)
    ax = ax_violin_plot.add_axes([0,0,1,1])
    ax.set_facecolor(backgroundColor)
    plt.tick_params(colors=textColor, which='both')
    ax.tick_params(axis='x', colors=backgroundColor)
    parts = ax.violinplot(violin_plot)

        # now change colors
    for pc in parts['bodies']:
            pc.set_facecolor(primaryColor)
            pc.set_edgecolor(secondaryColor)
            pc.set_alpha(1)

    
    plt.xlabel("Total Customers                                      Misspelling    ",fontsize=14,color=textColor)
    plt.ylabel("# Customers per store",  color=textColor, rotation='vertical', loc ='center',fontsize=14)
    
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey')
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')
    plt.figure(figsize=(10,10))

    return ax_violin_plot


def na_graphs (df, name):
    na, ax = plt.subplots()
    na.patch.set_facecolor(backgroundColor)
    ax.set_facecolor(backgroundColor)
    x = df['index']
    y = df['count']
    ax.barh(x, y, color=primaryColor)
    ax.tick_params(axis='y', colors=textColor)
    ax.tick_params(axis='x', colors=textColor)
        # Add cosmetics
    ax.spines['bottom'].set_color('lightgrey')
    ax.spines['top'].set_color('lightgrey') 
    ax.spines['right'].set_color('lightgrey')
    ax.spines['left'].set_color('lightgrey')
    plt.title(f"NAs on {name} data", color=textColor, fontsize =13)

    return na 


    