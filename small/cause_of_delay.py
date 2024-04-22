import pandas as pd

# Load the dataset
df = pd.read_csv('2023DecPA_cause_delay.csv')

# Columns of interest
delay_columns = ['CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY']

# Calculate total delay minutes for each column
total_delay_minutes = df[delay_columns].sum()

# Calculate total number of non-zero records for each column
non_zero_counts = df[delay_columns].astype(bool).sum()

# Display results
print("Total Delay Minutes per Column:")
print(total_delay_minutes)
print("\nTotal Number of Non-Zero Records per Column:")
print(non_zero_counts)
