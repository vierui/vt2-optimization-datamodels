# %%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

datadir = '/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/raw/data-load-becc.csv'

# Load the data
data = pd.read_csv(datadir, sep=';', parse_dates=['time'], index_col='time')

# Resample the data to weekly totals
weekly_data = data.resample('W').sum()

# Calculate mean and standard deviation for each week
stats = weekly_data.agg(['mean', 'std'])

# Correcting the way to access 'mean' values
mean_values = stats['load']['mean']  # Adjust this based on your data structure

# Identify representative weeks (simplified logic)
seasonal_weeks = {}
seasons = {'Winter': ('2022-12-01', '2023-02-28'), 'Spring': ('2023-03-01', '2023-05-31'),
           'Summer': ('2023-06-01', '2023-08-31'), 'Autumn': ('2023-09-01', '2023-11-30')}

for season, (start, end) in seasons.items():
    season_data = weekly_data.loc[start:end]
    # Here, adjust how you calculate the closest week using the correct mean reference
    closest_week = (season_data['load'] - mean_values).abs().idxmin()
    seasonal_weeks[season] = closest_week

print(seasonal_weeks)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Assuming data is already loaded and weekly_data is available
# Calculation of mean values and selection of representative weeks

# Calculate mean and standard deviation for each week globally
stats = weekly_data.agg(['mean', 'std'])

# Define the time frames for Winter, Spring_Autumn (combined), and Summer
seasons = {
    'Winter': ('2022-12-01', '2023-02-28'),
    'Spring_Autumn': ('2023-03-01', '2023-05-31', '2023-09-01', '2023-11-30'),
    'Summer': ('2023-06-01', '2023-08-31')
}

# Spring and Autumn data are combined
spring_data = weekly_data.loc['2023-03-01':'2023-05-31']
autumn_data = weekly_data.loc['2023-09-01':'2023-11-30']
spring_autumn_data = pd.concat([spring_data, autumn_data])

# Calculate the mean load for the combined Spring and Autumn
spring_autumn_mean = spring_autumn_data['load'].mean()

# Find the week closest to the mean of the combined Spring and Autumn
closest_spring_autumn_week = (spring_autumn_data['load'] - spring_autumn_mean).abs().idxmin()

# Identify representative weeks for each season
seasonal_weeks = {
    'Winter': weekly_data.loc['2022-12-01':'2023-02-28']['load'].sub(stats['load']['mean']).abs().idxmin(),
    'Spring_Autumn': closest_spring_autumn_week,
    'Summer': weekly_data.loc['2023-06-01':'2023-08-31']['load'].sub(stats['load']['mean']).abs().idxmin()
}

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 6))
weekly_data['load'].plot(ax=ax, label='Weekly Load Data')
for season, week in seasonal_weeks.items():
    ax.axvline(x=week, color='r', linestyle='--', label=f'{season} Week: {week.date()}')
plt.title('Weekly Load Data with Representative Weeks Highlighted for Combined Spring and Autumn')
plt.xlabel('Date')
plt.ylabel('Load')
plt.legend()
plt.grid(True)
plt.show()
# %%
# Define the date range for the plots (specific to the dataset used)
winter_data = pd.concat([
    weekly_data.loc['2023-12-01':'2023-12-31'],  # Previous year's December
    weekly_data.loc['2023-01-01':'2023-02-28']   # Current year's January and February
])
spring_autumn_data = pd.concat([
    weekly_data.loc['2023-03-01':'2023-05-31'],  # Spring months
    weekly_data.loc['2023-09-01':'2023-11-30']   # Autumn months
])
summer_data = weekly_data.loc['2023-06-01':'2023-08-31']  # Summer months
# Calculate mean values for annotation
winter_mean = winter_data['load'].mean()
spring_autumn_mean = spring_autumn_data['load'].mean()
summer_mean = summer_data['load'].mean()

# Prepare the subplots horizontally
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Winter plot
axs[0].plot(winter_data.index, winter_data['load'], label='Winter Weekly Load')
axs[0].axvline(x=seasonal_weeks['Winter'], color='r', linestyle='--', label=f'Representative Week: {seasonal_weeks["Winter"].date()}')
axs[0].set_title('Winter Load Data')
axs[0].set_ylabel('Load')
axs[0].legend()
axs[0].grid(True)
axs[0].text(0.01, 0.95, f'Mean Load: {winter_mean:.2f}', transform=axs[0].transAxes, verticalalignment='top')

# Spring_Autumn plot
axs[1].plot(spring_autumn_data.index, spring_autumn_data['load'], label='Spring_Autumn Weekly Load')
axs[1].axvline(x=seasonal_weeks['Spring_Autumn'], color='r', linestyle='--', label=f'Representative Week: {seasonal_weeks["Spring_Autumn"].date()}')
axs[1].set_title('Spring and Autumn Load Data')
axs[1].set_ylabel('Load')
axs[1].legend()
axs[1].grid(True)
axs[1].text(0.01, 0.95, f'Mean Load: {spring_autumn_mean:.2f}', transform=axs[1].transAxes, verticalalignment='top')

# Summer plot
axs[2].plot(summer_data.index, summer_data['load'], label='Summer Weekly Load')
axs[2].axvline(x=seasonal_weeks['Summer'], color='r', linestyle='--', label=f'Representative Week: {seasonal_weeks["Summer"].date()}')
axs[2].set_title('Summer Load Data')
axs[2].set_ylabel('Load')
axs[2].legend()
axs[2].grid(True)
axs[2].text(0.01, 0.95, f'Mean Load: {summer_mean:.2f}', transform=axs[2].transAxes, verticalalignment='top')

# Set major and minor locators and formatters for x-axis
for ax in axs:
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('W%V'))
    ax.xaxis.set_tick_params(rotation=45, which='both')

plt.tight_layout()
plt.show()
# %%
import pandas as pd

# File paths
load2_winter_path = '/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/load2-autumn_spring.csv'
utility_path = '/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/raw/utility.csv'
output_path = '/Users/rvieira/Documents/Master/vt1-energy-investment-model/data/processed/load2-autumn_spring.csv'

# Read the load2-winter.csv
load2_df = pd.read_csv(load2_winter_path, sep=';', dtype={'time': str, 'value': float})

# Read the utility.csv and ensure proper format
utility_df = pd.read_csv(utility_path, sep=';', dtype={'time': str, 'value': str})

# Convert scientific notation in 'value' to float (removing scientific notation)
utility_df['value'] = utility_df['value'].apply(lambda x: float(x))

# Replace the 'value' column in load2_df with the 'value' column from utility_df
load2_df['value'] = utility_df['value'].values

# Save the updated DataFrame to a new CSV file
load2_df.to_csv(output_path, sep=';', index=False, float_format='%.3f')

print(f"Updated file saved to: {output_path}")

# %%
