import os
import pandas as pd
from datetime import datetime
import ipdb

# Directory containing the CSV files
directory = '/mlwell-data2/dafna/PACEHD_for_ssl_paper'

# Output text file
output_file = 'PACE_start_times.txt'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Construct full file path
            file_path = os.path.join(directory, filename)
            
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the first timestamp from the second row of the first column
            timestamp = df.iloc[1, 0]
            
            # Convert the timestamp to human-readable format
            readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            # Write the result to the output file
            outfile.write(f"{filename}: {readable_time}\n")

print("Processing complete. Check the output.txt file for results.")