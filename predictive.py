import pandas as pd
import numpy as np

class TurbineDataStream():
    """This class simulates a wind turbine that generates data
    in a continuous stream and calculates kpis on the fly.
    """
    
    def __init__(self, rel_path_to_turbine_data):
        """Loads the entire turbine data set from path
        in order to simulate the data stream.
        """
        self.turbine_data = pd.read_csv(
                rel_path_to_turbine_data, sep=","
        )
        self.turbine_data.dropna(
            axis=0, subset=["ActivePower"], inplace=True
        )
        self.data = {
            'Unnamed: 0': None,
            'ActivePower_t0': None,
            'ActivePower_t1': None,
            'ActivePower_t2': None,
            'ActivePower_t3': None,
            'ActivePower_t4': None,
            'ActivePower_t5': None,
            'ActivePower_t6': None,
            'avg_WindSpeed': None
        }
        
    def __len__(self):
        """Returns the number of data points."""
        len_ = self.turbine_data.shape[0]
        
        return len_
    
    def mean_online(self, mean_old, data_new, alpha=0.1):
        """Calculates the mean in an online fashion."""
        if not mean_old: mean_new = data_new
        else: # if data exists: EWMA
            if pd.isna(data_new): mean_new = mean_old
            else: mean_new = mean_old + alpha * (data_new - mean_old)
    
        return mean_new
    
    def pred_active_power(self, t):
        """Predicts ActivePower for the next six timestamps after t.
        To-do: Use a time series model instead of random noise.
        """
        active_power_t0 = self.turbine_data.loc[t, 'ActivePower']
        active_power_t1 = active_power_t0 + np.random.normal(0, 50)
        active_power_t2 = active_power_t1 + np.random.normal(0, 50)
        active_power_t3 = active_power_t2 + np.random.normal(0, 50)
        active_power_t4 = active_power_t3 + np.random.normal(0, 50)
        active_power_t5 = active_power_t4 + np.random.normal(0, 50)
        active_power_t6 = active_power_t5 + np.random.normal(0, 50)
        
        return [
            active_power_t0,
            active_power_t1, active_power_t2, active_power_t3,
            active_power_t4, active_power_t5, active_power_t6
        ]
    
    def __iter__(self):
        """Yields the data and kpis of the stream one by one.
        """
        for t in self.turbine_data.index:
            
            self.data['Unnamed: 0'] = self.turbine_data.loc[t, 'Unnamed: 0']
            ActivePower_t0_to_t6 = self.pred_active_power(t)
            self.data.update(
                zip(
                    ['ActivePower_t0',
                     'ActivePower_t1', 'ActivePower_t2', 'ActivePower_t3',
                     'ActivePower_t4', 'ActivePower_t5', 'ActivePower_t6'],
                    ActivePower_t0_to_t6
                )
            )
            self.data['avg_WindSpeed'] = self.mean_online(
                    self.data['avg_WindSpeed'],
                    self.turbine_data.loc[t, 'WindSpeed']
            )
            
            yield self.data

def cut_timeframe_from_df(df, start_date, stop_date, start_time, stop_time):
  analysis_start_datetime = start_date + " " + start_time + "+00:00"
  analysis_stop_datetime = stop_date + " " + stop_time + "+00:00"
  selected_df = df[analysis_start_datetime:analysis_stop_datetime]
  return selected_df, analysis_start_datetime, analysis_stop_datetime

