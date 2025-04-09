import pandas as pd

#read files
def load_csvs():
    ausgrid2010 = pd.read_csv(r'ausgrid2010\ausgrid2010.csv', skiprows=[0])
    ausgrid2011 = pd.read_csv(r'ausgrid2011\ausgrid2011.csv', skiprows=[0])
    ausgrid2012 = pd.read_csv(r'ausgrid2012\ausgrid2012.csv', skiprows=[0])
    
    return ausgrid2010,ausgrid2011, ausgrid2012

def prepare_dataframe(df, fileFormat = 1):
    df = df[df['Consumption Category'] == 'GC'].drop(columns = ['Consumption Category', 'Generator Capacity', 'Postcode'])
    if 'Row Quality' in df.columns:
        df = df.drop(columns = ['Row Quality'])
        
    
    # Melt dataframe to long format
    df_melt = df.melt(id_vars=['Customer', 'date'], var_name='time')
    
    # Combine 'date' and 'time' columns
    
    if fileFormat:
        df_melt['datetime'] = pd.to_datetime(df_melt['date'] + ' ' + df_melt['time'], format='%d/%m/%Y %H:%M')
    else:
        df_melt['datetime'] = pd.to_datetime(df_melt['date'] + ' ' + df_melt['time'], format='%d-%b-%y %H:%M')

    
    # You may want to sort values by 'Customer' and 'datetime'
    df_melt = df_melt.sort_values(by=['Customer', 'datetime'])
    
    # You can drop 'date' and 'time' columns if you no longer need them
    df_melt = df_melt.drop(columns=['date', 'time'])
    
    # Reset the index
    df_melt = df_melt.reset_index(drop=True)
    df = df_melt.pivot(index='Customer', columns='datetime', values='value')
    df = df.T.sort_index()
    
    return df

def prepare_dfs(ausgrid2010,ausgrid2011, ausgrid2012):
    ausgrid2010 = prepare_dataframe(ausgrid2010, 0)
    ausgrid2011 = prepare_dataframe(ausgrid2011)
    ausgrid2012 = prepare_dataframe(ausgrid2012)
    
    return ausgrid2010,ausgrid2011, ausgrid2012

def concat_dfs(ausgrid2010,ausgrid2011, ausgrid2012):
    ausgrid_DF = pd.concat((ausgrid2010,ausgrid2011,ausgrid2012), axis = 0)
    ausgrid_DF = ausgrid_DF.ffill()
    
    return ausgrid_DF

def main():
    
    ausgrid2010,ausgrid2011, ausgrid2012 = load_csvs()
    
    ausgrid2010,ausgrid2011, ausgrid2012 = prepare_dfs(ausgrid2010,ausgrid2011, ausgrid2012)
    
    ausgrid_DF = concat_dfs(ausgrid2010,ausgrid2011, ausgrid2012)
    
    return ausgrid_DF
    




