import requests
import os
import sys
from dotenv import load_dotenv
from datetime import date, timedelta

# Determine the path to .env.local relative to this script
script_dir = os.path.abspath(os.path.dirname(__file__))
# Go up two directories from script_dir (e.g., scripts/forecasting -> scripts -> project_root)
project_root = os.path.normpath(os.path.join(script_dir, '..', '..'))
dotenv_file_path = os.path.join(project_root, '.env.local')

# Load environment variables from the determined path
load_dotenv(dotenv_path=dotenv_file_path)

token = os.getenv('RENEWABLES_TOKEN')
if token is None:
    print("Error: RENEWABLES_TOKEN environment variable not set.")
    sys.exit(1)

api_base = 'https://www.renewables.ninja/api/'

s = requests.session()
# Send token header with each request
s.headers = {'Authorization': 'Token ' + token}

# --- PV example ---
url = api_base + 'data/weather'

# --- WIND example ---
# url = api_base + 'data/weather'

args = {
    'lat': 46.2312,
    'lon': 7.3589,
    'date_from': '2015-01-01',
    'date_to': '2024-12-31',
    'dataset': 'merra2',
    # 'capacity': 1.0,
    # 'system_loss': 0.1,
    # 'tracking': 0,
    # 'tilt': 35,
    # 'azim': 180,
    'format': 'csv',
    # include local time column and header comments:
    # 'local_time':        'true',
    'header':            'true',
    # the four weather vars you want:
    'var_t2m':           'true',
    'var_prectotland':   'true',
    'var_swgdn':         'true',
    'var_cldtot':        'true',
}

# --- Start of new section for chunking and fetching ---
original_date_from_str = args['date_from']
original_date_to_str = args['date_to']

try:
    overall_start_date = date.fromisoformat(original_date_from_str)
    overall_end_date = date.fromisoformat(original_date_to_str)
except ValueError:
    print(f"Error: Invalid date format for date_from ('{original_date_from_str}') or date_to ('{original_date_to_str}'). Please use YYYY-MM-DD.")
    sys.exit(1)

# Define output directory relative to the project root
output_dir = os.path.join(project_root, 'data', 'renewables')
os.makedirs(output_dir, exist_ok=True)
file_name = 'weather_data.csv'
file_path = os.path.join(output_dir, file_name)

if overall_start_date > overall_end_date:
    print(f"Error: date_from ('{overall_start_date}') is after date_to ('{overall_end_date}'). No data to fetch.")
    with open(file_path, 'w', encoding='utf-8') as f:
        pass # Create an empty file
    print(f"No data to fetch for the given date range. Empty file created at {file_path}")
    sys.exit(0)

all_chunks_successful = True
is_first_chunk = True
data_written = False # To track if any data was actually written from API responses

print(f"Preparing to fetch data from {original_date_from_str} to {original_date_to_str} in chunks constrained by a 1-year period.")

with open(file_path, 'w', encoding='utf-8') as f:
    current_loop_start_date = overall_start_date
    while current_loop_start_date <= overall_end_date:
        try:
            # Calculate end date for this chunk: current_loop_start_date + 1 year - 1 day
            # This ensures the period (date_from to date_to inclusive) is at most 1 year.
            one_year_later_date = current_loop_start_date.replace(year=current_loop_start_date.year + 1)
        except ValueError:  # Handles case e.g. date(2020, 2, 29).replace(year=2021) -> date(2021,2,28)
            one_year_later_date = current_loop_start_date.replace(year=current_loop_start_date.year + 1, day=28)
        
        chunk_end_date_ideal = one_year_later_date - timedelta(days=1)
        current_chunk_actual_end_date = min(chunk_end_date_ideal, overall_end_date)

        args_for_chunk = args.copy() 
        args_for_chunk['date_from'] = current_loop_start_date.strftime('%Y-%m-%d')
        args_for_chunk['date_to'] = current_chunk_actual_end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching data for chunk: {args_for_chunk['date_from']} to {args_for_chunk['date_to']}")
        r = s.get(url, params=args_for_chunk)

        if r.status_code == 200:
            csv_text = r.text.strip() # Remove leading/trailing whitespace, including final newline for consistency
            if csv_text: # Proceed only if there's some text content
                if is_first_chunk:
                    f.write(csv_text + '\n') # Write the first chunk (includes headers) and ensure a newline
                    data_written = True
                    is_first_chunk = False
                else:
                    lines = csv_text.splitlines()
                    # Renewables.ninja API: 3 metadata lines starting with '#', then 1 data header line
                    num_header_lines_to_skip = 4 
                    if len(lines) > num_header_lines_to_skip:
                        data_lines = lines[num_header_lines_to_skip:]
                        content_to_append = '\n'.join(data_lines)
                        if content_to_append: # Ensure there's actual data after joining
                            f.write(content_to_append + '\n') # Append data lines and ensure a newline
                            data_written = True
                    # If len(lines) <= num_header_lines_to_skip, it means only headers or less for this chunk.
            # If csv_text is empty after strip(), do nothing for this chunk.
        else:
            print(f"Failed to retrieve data for chunk {args_for_chunk['date_from']} to {args_for_chunk['date_to']}.")
            print(f"Status code: {r.status_code}, Response: {r.text[:500].strip()}...") # Print first 500 chars of error
            all_chunks_successful = False
            break # Stop processing further chunks

        current_loop_start_date = current_chunk_actual_end_date + timedelta(days=1)

# After the loop and closing the file
if all_chunks_successful:
    if data_written:
        print(f"Data successfully saved to {file_path}")
    else:
        print(f"All API calls were successful, but no actual data content was written to {file_path}. The file might be empty or contain only headers if the first chunk was empty too.")
else:
    print(f"Data fetching failed for some chunks. Partial data might be in {file_path}.")
# --- End of new section ---