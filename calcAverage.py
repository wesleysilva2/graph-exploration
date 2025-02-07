import os
import pandas as pd
import pickle
import re

def process_date_folder(date_str: str) -> dict:
    date = date_str.split("-")
    file_path = f"./sunt/{date_str}/output/trips_time-series_{date[2]}-{date[1]}-{date[0]}_{date[2]}-{date[1]}-{date[0]}.parquet"
    if not os.path.exists(file_path):
        print(f"File not found for date: {file_path}")
        return None
    
    df = pd.read_parquet(file_path)
    time_per_stop = {}  # array where [total time, number of trips]
    
    print(f"Processing {date_str}...")
    i = 0
    for index, row in df.iterrows():
        if index == 0:
            continue

        previous_row = df.loc[index-1]
    
        if previous_row["linha_atend"] != row["linha_atend"]:
            continue

        stop_pair = (previous_row["stop_id"], row["stop_id"])

        if stop_pair not in time_per_stop:
            time_per_stop[stop_pair] = [0, 0]

        time_per_stop[stop_pair][0] += row["tempo_total"]
        time_per_stop[stop_pair][1] += 1

        if index / len(df.index) * 100 > i: 
            i += 1
            print(f"\r{i}% of {len(df.index)} rows for {date_str}", end='')
    
    return time_per_stop


base_path = "./sunt"
date_folders = [f for f in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, f)) 
                and re.match(r'\d{4}-\d{2}-\d{2}', f)
                ]
date_folders.sort()

# Process all dates and combine results
combined_averages = {}


for index, date_folder in enumerate(date_folders):
    print(f"{index}/{len(date_folders)}")
    result = process_date_folder(date_folder)

    if result == None:
        continue
    
    with open(f"./output/averages_{date_folder}.pkl", "wb") as f:
        pickle.dump(result, f)
    print(f"\nSaved results for {date_folder}")
    
    print(f"Combining {date_folder}...", end="\n\n")
    for stop_pair, (total_time, count) in result.items():
        if stop_pair not in combined_averages:
            combined_averages[stop_pair] = [0, 0]
        combined_averages[stop_pair][0] += total_time
        combined_averages[stop_pair][1] += count

print(f"Saving combined averages...")
with open(f"./output/combined_sum_amount.pkl", "wb") as f:
    pickle.dump(combined_averages, f)


final_averages = {
    stop_pair: total_time / count 
    for stop_pair, (total_time, count) in combined_averages.items()
}

final_averages