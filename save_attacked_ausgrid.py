import matplotlib.pyplot as plt
import random
import pandas as pd
import load_ausgrid
import sys
sys.path.insert(1, r'D:\Initial_workspace')
import attackTypes
def plot_sample_days(attDF, df, attackType):
    
    # Define number of customers and days
    n_customers = 1
    n_days = 1
    
    # Randomly select 10 customers
    customer = random.sample(list(attDF.columns), n_customers)
    customer = 100
    # Randomly select 10 days
    day = random.sample(list(attDF.index.date), n_days)
    day = attDF.index.date[15]
    fig, axs = plt.subplots(n_customers, figsize=(20, 3))
    
    # For each customer
    # Select data for current customer
    df_customer = attDF[customer]
    df_original = df[customer]
    # For each day
    # Select data for current day
    df_day = df_customer[df_customer.index.date == day]
    original_day = df_original[df_original.index.date == day]
    # Plot data
    if attackType == 'ieee':
        attackType = 13
    axs.plot([str(i)[:-3] for i in list(attDF.index.time)[0:48]], df_day.values, label= f'after attack\n{attackType}', color = 'red',linewidth = 9)
    axs.plot([str(i)[:-3] for i in list(df.index.time)[0:48]], original_day.values, label='original values', color = 'blue', alpha = 0.8 , linewidth = 9)
    # axs.legend(fontsize = 20)

    axs.set_xticks([])
    axs.set_yticks([])
    # Set title and labels
    axs.tick_params(axis='x', rotation=90)
    # axs.set_title('Customer ' + str(customer))
    # axs.set_xlabel('Time')
    # axs.set_ylabel('Value')
    
    
    # Display plot
    # plt.title(f'attack {attackType}')
    plt.tight_layout()
    plt.show()
    


attacks = ['ieee', *range(0, 13)]

def save_dfs():
    ausgrid_df = load_ausgrid.main()
    
    ausgrid_df.to_hdf('ausgrid_attacked.h5', key = 'original')
    for attack in attacks:
        
        ausgrid_att = attackTypes.attack(attack, ausgrid_df.copy(), 0)
        
        ##save to hd5 file
        
        ausgrid_att.to_hdf('ausgrid_attacked.h5', key = f'attack{attack}')


    
def plot_sample():
    for attack in ['original', *attacks]:
        if attack == 'original':
            ausgrid_original = pd.read_hdf("ausgrid_attacked.h5", key = 'original')
        else:
            ausgrid_att = pd.read_hdf("ausgrid_attacked.h5", key = f'attack{attack}')
            plot_sample_days(ausgrid_att, ausgrid_original, attack)


