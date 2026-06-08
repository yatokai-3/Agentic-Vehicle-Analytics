import pandas as pd
from pathlib import Path

import pandas as pd
from pathlib import Path


def process_vehicle_data():

    all_data = []

    month_names = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ]

    project_path = Path(".")

    # Loop through state folders
    for state_folder in project_path.iterdir():

        if not state_folder.is_dir():
            continue

        state_name = state_folder.name

        # Find all CSV files inside state folder
        for file in state_folder.glob("*.csv"):

            try:

                print(f"Processing: {file}")

                df = pd.read_csv(file, header=None)

                # Remove accidental numeric header row
                if str(df.iloc[0, 0]) == "0":
                    df = df.iloc[1:]

                # Standard columns
                df.columns = [
                    'Year',
                    'State',
                    'Rto_name',
                    'Index',
                    'Type',
                    *month_names,
                    'Total'
                ]

                # Detect datatype from filename
                filename = file.stem.lower()

                if "fuel" in filename:
                    data_type = "fuel"

                elif "category" in filename:
                    data_type = "category"

                elif "class" in filename:
                    data_type = "class"

                else:
                    data_type = "unknown"

                df["Data_Type"] = data_type

                # Keep useful columns
                df = df[
                    ['Year', 'State', 'Rto_name', 'Type', 'Data_Type']
                    + month_names
                ]

                # Convert wide → long
                df_long = df.melt(
                    id_vars=[
                        'Year',
                        'State',
                        'Rto_name',
                        'Type',
                        'Data_Type'
                    ],
                    value_vars=month_names,
                    var_name='Month',
                    value_name='Count'
                )

                # Clean data
                df_long['Year'] = pd.to_numeric(
                    df_long['Year'],
                    errors='coerce'
                )

                df_long['Count'] = (
                    df_long['Count']
                    .astype(str)
                    .str.replace(',', '')
                )

                df_long['Count'] = pd.to_numeric(
                    df_long['Count'],
                    errors='coerce'
                ).fillna(0).astype(int)

                df_long = df_long[df_long['Year'].notna()]

                all_data.append(df_long)

                print(f"✓ Done: {file.name}")

            except Exception as e:
                print(f"✗ Error in {file}: {e}")

    # Combine everything
    master_df = pd.concat(all_data, ignore_index=True)

    # Save final file
    master_df.to_csv("master_vehicle_data.csv", index=False)

    # print("\nDone!")
    # print(master_df.head())

    return master_df


if __name__ == "__main__":

    df = process_vehicle_data()
