import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files
cengen_neuron_df = pd.read_csv('cengen_neuron_prediction.csv')
neuron_connect_df = pd.read_csv('NeuronConnect.csv')

# Calculate the total number of connections for each neuron
# Sum up all the connections (Nbr) for each neuron in both Neuron 1 and Neuron 2
neuron_connections = pd.concat([
    neuron_connect_df[['Neuron 1', 'Nbr']].rename(columns={'Neuron 1': 'neuron'}),
    neuron_connect_df[['Neuron 2', 'Nbr']].rename(columns={'Neuron 2': 'neuron'})
])

# Group by neuron and sum the 'Nbr' values to get total connections per neuron
total_connections = neuron_connections.groupby('neuron')['Nbr'].sum().reset_index()

# Merge the connection data with the CeNGEN neuron prediction data based on neuron name
# Use a left join to keep all neurons from cengen_neuron_df, even if they don't appear in neuron_connections
merged_df = pd.merge(cengen_neuron_df, total_connections, on='neuron', how='left')

# Fill NaN values in 'Nbr' (which occur when a neuron has no connections) with 0
merged_df['Nbr'] = merged_df['Nbr'].fillna(0)

# Plot a scatter plot with connections on the x-axis and age on the y-axis
plt.figure(figsize=(10, 6))
plt.scatter(merged_df['Nbr'], merged_df['CeNGEN_BitAge_Prediction'], alpha=0.5)

# Add labels and title
plt.xlabel('Number of Connections (Nbr)')
plt.ylabel('Neuron Age (CeNGEN_BitAge_Prediction)')
plt.title('Neuron Age vs Number of Connections')
plt.grid(True)

# Show the plot
plt.show()
