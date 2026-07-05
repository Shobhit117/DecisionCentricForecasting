import numpy as np
import pandas as pd

DATASET_FILE_PATH = 'data/m5-forecasting-accuracy/sales_train_evaluation.csv'
CALENDAR_FILE_PATH = 'data/m5-forecasting-accuracy/calendar.csv'

def process_m5_calendar(calendar : pd.DataFrame) -> pd.DataFrame:
    calendar['day'] = calendar['d'].str.split('_').str[1].astype(int)
    calendar = calendar[['day', 'wday', 'month', 'year', 'date', 'event_name_1', 'event_type_1']].rename(columns={'event_type_1': 'event_type', 'event_name_1': 'event_name'})
    return calendar

def read_m5_data():
    df = pd.read_csv(DATASET_FILE_PATH)
    calendar = process_m5_calendar(pd.read_csv(CALENDAR_FILE_PATH))
    return df, calendar

def process_m5_data(df : pd.DataFrame, calendar : pd.DataFrame = None) -> pd.DataFrame:
    ts_id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = df.melt(id_vars=ts_id_cols, var_name='day', value_name='demand')
    df['day'] = df['day'].str.split('_').str[1].astype(int)
    if calendar is not None:
        df = pd.merge(df, calendar, on='day', how='left')
    return df

def m5_chunk_processor(chunk_size : int = 100):
    reader = pd.read_csv(DATASET_FILE_PATH, chunksize=chunk_size)
    calendar = process_m5_calendar(pd.read_csv(CALENDAR_FILE_PATH))
    for chunk in reader:
        yield process_m5_data(chunk, calendar)

def load_m5_data(chunk_size : int = 100, num_chunks : int = 1):
    m5_data = []
    for i, chunk in enumerate(m5_chunk_processor(chunk_size)):
        m5_data.append(chunk)
        if i == num_chunks - 1:
            break
    m5_data = pd.concat(m5_data)
    return m5_data

def generate_synthetic_lead_times(store_list : list[str]
                                , seed : int = 0):
    np.random.seed(seed)
    num_entries = 100
    df = pd.DataFrame({'store_id': store_list * num_entries})
    df['lead_time'] = np.random.randint(1, 6)
    return df

