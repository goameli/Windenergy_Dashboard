import pandas as pd
from pathlib import Path


def get_data(folder_data=Path(".", "data"), data_name="Turbine_Data.csv"):
    """Get data from defined Path & defined data name

    Args:
        folder_data ([type], optional): [description]. Defaults to Path(".", "data").
        data_name (str, optional): [description]. Defaults to "Turbine_Data.csv".

    Returns:
        pd.DataFrame: Returns Dataframe back.
    """
    path_alarm_csv_file = Path(folder_data, data_name)

    df = pd.read_csv(path_alarm_csv_file)

    return df


def make_datetime_from_columns(df, input_column, output_column="timestamp", time_format="%Y.%m.%d %H:%M:%S"):
    """Take a input column and make it timestamp index (drop old columns). Inplace operation.

    Args:
        df (pd.DataFrame): Dataframe with given Columns
        input_column (str): Column name that should be timestamp index
        output_column (np.datetime, optional): Columnname of new column. Defaults to "timestamp".
        time_format (str, optional): Timeformat for conversion (see datetime strptime-format-codes). Defaults to ""%Y.%m.%d %H:%M:%S"".

    Returns:
        pd.DataFrame: Returns Dataframe back
    """
    df[output_column] = df.pop(input_column).astype(str)    
    df[output_column] = pd.to_datetime(df[output_column], format=time_format)
    df.set_index(output_column, inplace=True)

    return df