"""In this Python coding, the Pandas, Matplotlib,numpy
and seaborn are used """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function for reading and preprocessing data on greenhouse gas emissions.

def greenhouse_gas_emissions(csv_file): 
    """Read and preprocess data on greenhouse gas emissions."""
    df = pd.read_csv("Total greenhouse gas emissions (kt of CO2 equivalent).csv", skiprows=4)
    # Remove any columns relating to country codes and indicators that are no longer required.
    df = df.loc[:, ~df.columns.isin(["Country Code", "Indicator Name", "Indicator Code"])]
    # Remove any columns containing NaN values.
    df = df.dropna(axis=1, how='all')
    # Choose columns from the year 1995 to 2005.
    df = df[['Country Name'] + list(map(str, range(1995, 2006)))]
    # Create an Excel file with the processed data.
    df.to_excel('data.xlsx', index=True)
    df_transposed = df.set_index('Country Name').T
    #Removw the NaN values in trasnsposed data
    df_transposed = df_transposed.dropna(how='all')
    #Transpose the data and save it to an Excel file.
    df_transposed.to_excel('transposed_data.xlsx', index=True)  
   
    return df, df_transposed 
    
df,df_transposed = greenhouse_gas_emissions('Total greenhouse gas emissions (kt of CO2 equivalent).csv')

# Function to read and preprocess csv file
def aff_value(csv_file):
    """Read and preprocess csv file for agriculture, forestry, and fishing value added."""

    df1 = pd.read_csv("Agriculture, forestry, and fishing, value added (% of GDP).csv",skiprows=4)
    df1 = df1.loc[:, ~df1.columns.isin(["Country Code","Indicator Name","Indicator Code"])]
    df1 = df1.dropna(axis=1, how = 'all')
    df1 = df1[['Country Name'] + list(map(str, range(1995, 2006)))]
    df1.to_excel('raw data.xlsx', index=True)
    df1_transposed = df1.set_index('Country Name').T
    df1_transposed = df1_transposed.dropna(how ='all')
    df1_transposed.to_excel('raw_transposed.xlsx', index=True)
    
    return df1,df1_transposed

df1,df1_transposed = aff_value("Agriculture, forestry, and fishing, value added (% of GDP).csv")
#Function for summary statistics 
def summary_statistics(dataframe, indicators, countries):
    """Explore and cross-compare summary statistics for selected indicators and countries."""
    for indicator in indicators:
        print(f"\nSummary Statistics for {indicator}:")
        for country in countries:
            if country in dataframe.columns:
                country_data = dataframe[country]
                print(f"\n{country}:")
                
                # Method 1: Using pandas describe
                print("Using Pandas describe:")
                print(country_data.describe())

                # Method 2: Using NumPy median
                median_value = np.median(country_data)
                print(f"Median: {median_value}")

                # Method 3: Using NumPy standard deviation
                std_dev = np.std(country_data)
                print(f"standard deviation: {std_dev}")
            else:
                print(f"\n{country} not found in the dataset.")
# Example usage
selected_indicators = ['Total greenhouse gas emissions (kt of CO2 equivalent)']
selected_countries = ['Germany', 'China', 'United Kingdom', 'India']

summary_statistics(df_transposed, selected_indicators, selected_countries)

# Plot a bar chart - 1 
def bar_chart(df_transposed, columns_to_compare, title, ylabel):
   
   """Plot a bar chart for greenhouse gas emissions."""
 
   columns_to_compare = ['Germany', 'China', 'Euro area','United Kingdom','India']  # Replace with actual country names

    
   plt.figure(figsize=(10, 6))
   colors = ['skyblue', 'pink', 'lightgreen','red','orange']
   df_transposed[columns_to_compare].plot(kind='bar', color=colors, width=0.8, edgecolor='black')

   plt.title('Greenhouse Gas Emissions by Country (1995-2005)')
   plt.xlabel('year')
   plt.ylabel('Emissions (kt of CO2 )')
   plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))
   plt.xticks(rotation=32, ha='right')
   plt.grid(axis='y', linestyle='-', alpha=0.8)
   plt.tight_layout()
   plt.show()
   
# Plot a bar chart - 2
def bar2_chart(df1_transposed, columns_to_compare, title, ylabel):
    """Plot a line chart for agriculture, forestry, and fishing value added."""
    columns_to_compare = ['Germany', 'China', 'Euro area','United Kingdom','India']

    plt.figure(figsize=(10, 6), dpi=300)
    colors = ['skyblue', 'lightcoral', 'lightgreen','purple','green']
    df1_transposed[columns_to_compare].plot(kind='bar', color=colors, width=0.8, edgecolor='black')

    plt.title('Agriculture, Forestry, and Fishing Value Added (% of GDP) (1995-2005)')
    plt.xlabel('Year')
    plt.ylabel('Value Added (% of GDP)')
    plt.legend(title='Country', loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=35, ha='right')
    plt.grid(axis='y', linestyle='-', alpha=0.7)

    plt.tight_layout()
    plt.show()


#Line plot - 1
def line_plot(df, selected_countries, selected_years):
    """Plot a line chart for greenhouse gas emissions."""
    df_selected_years = df[df['Country Name'].isin(selected_countries)][['Country Name'] + selected_years]

    # Plot a separate line for each selected year for the specified countries
    plt.figure(figsize = (12, 6))
    for year in selected_years:
        plt.plot(df_selected_years['Country Name'], df_selected_years[year], marker='o', linestyle='-', label=year)

    plt.xlabel('Country')
    plt.ylabel('Greenhouse Gas Emissions (kt of CO2 equivalent)')
    plt.title('Greenhouse Gas Emissions for Selected Countries (1996-2004)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.show()

#Line plot - 2
def line2_plot(data, selected_countries, selected_years):
    """Plot a line chart for agriculture, forestry, and fishing value added."""
    data_selected_years = data[data['Country Name'].isin(selected_countries)][['Country Name'] + selected_years]

    # Plot a separate line for each selected year for the specified countries
    plt.figure(figsize=( 12, 6 ))
    for year in selected_years:
        plt.plot(data_selected_years['Country Name'], data_selected_years[year], marker='o', linestyle='-', label=year)

    plt.xlabel('Country')
    plt.ylabel('Agriculture, Forestry, and Fishing Value Added (% of GDP)')
    plt.title('Agriculture, Forestry, and Fishing Value Added (% of GDP) for Selected Countries (1995-2005)')
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='-', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Example usage
columns_to_compare_gg = ['Germany', 'China', 'Euro area', 'United Kingdom', 'India']
bar_chart(df_transposed, columns_to_compare_gg, 'Greenhouse Gas Emissions by Country (1995-2005)', 'Emissions (kt of CO2 equivalent)')

columns_to_compare_aff = ['Germany', 'China', 'Euro area', 'United Kingdom', 'India']
bar2_chart(df1_transposed, columns_to_compare_aff, 'Agriculture, Forestry, and Fishing Value Added (% of GDP) (1995-2005)', 'Value Added (% of GDP)')

line_plot(df, ['Germany', 'China', 'Euro area', 'United Kingdom', 'India'], list(map(str, range(1996, 2005))))
line2_plot(df1, ['Germany', 'China', 'Euro area', 'United Kingdom', 'India'], list(map(str, range(1996, 2005))))

#Function to read climate change data
def read_climate_data(csv_file):
    """Read climate change data from CSV file."""
    return pd.read_csv(csv_file, skiprows=3)
# Function to extract and process environmental indicator data for China
def extract_and_process_data(data, country, indicator_name, indicator_code, start_year, end_year):
    """Extract and process environmental indicator data for a specific country."""
    country_data = data.loc[
        (data['Country Name'] == country) & (data['Indicator Code'] == indicator_code),
        start_year:end_year
    ].transpose()
    country_data.columns = [indicator_name]
    country_data = country_data.iloc[4:].astype(float)
    return country_data

def combine_dataframes(dataframes):
    """Combine a list of dataframes into a single dataframe."""
    return pd.concat(dataframes, axis=1)

def fill_missing_values(dataframe):
    """Fill missing values in a dataframe with zeros."""
    return dataframe.fillna(0)

def generate_correlation_heatmap(dataframe):
    """Generate and display a correlation heatmap for a dataframe."""
    plt.figure(figsize=(8, 5))
    sns.heatmap(dataframe.corr(), annot=True, cmap='seismic')  # Change the cmap parameter to 'seismic'
    plt.title('Correlation Heatmap China')
    plt.show()

# Read climate change data
climate_change = read_climate_data('climate change.csv')

# Define indicators for China
indicators_china = {
    'Population, total': 'SP.POP.TOTL',
    'Nitrous oxide emissions (% change from 1990)': 'EN.ATM.NOXE.ZG',
    'CO2 emissions (kg per 2015 US$ of GDP)': 'EN.ATM.CO2E.KD.GD',
    'Electric Power Consumption': 'EG.USE.PCAP.KG.OE',
    'Access to electricity (% of population)': 'EG.ELC.ACCS.ZS'
}

# Extract and process data for China
china_dataframes = []

for indicator_name, indicator_code in indicators_china.items():
    data = extract_and_process_data(
        climate_change, 'China', indicator_name, indicator_code, '1960', '2005'
    )
    china_dataframes.append(data)

# Combine the dataframes for China
china_data_combined = combine_dataframes(china_dataframes)

# Fill missing values with zeros
china_data_combined_filled = fill_missing_values(china_data_combined)

# Print correlation matrix
print("\nCorrelation Matrix for China:")
print(china_data_combined_filled.corr())

# Generate and display correlation heatmap with a different colormap
generate_correlation_heatmap(china_data_combined_filled)