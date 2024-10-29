import pandas as pd
from data_builder_for_model import *
from sample_parser import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import ranksums

def get_united_data_as_df():
    samples_series = AllSampleSeries(atlas=True, zimmer=False, flavell=False)
    samples = samples_series.get_all_samples(require_annotation=True)

    all_names_np = get_all_names_np(samples)

    table, _, _ = build_input_output_table_splited(samples, all_names_np,smooth_data=False)

    table = np.vstack(table)
    print('Created table with shape:', table.shape)
    return pd.DataFrame(table, columns=all_names_np)

def compute_correlation_matrix(data, type):
    df = data.reindex(sorted(data.columns), axis=1)
    correlation_matrix = df.corr(method=type, min_periods=1)
    print('Computed correlation matrix type:', type)
    return correlation_matrix

def plot_correlation_heapmap(corr_mat, type, neurons_age_df=None):
    plt.figure(figsize=(12, 10))
    plt.title('Correlation Heatmap (' + type + ')')
    if neurons_age_df is not None:
        heatmap_neurons = corr_mat.index
        sorted_neurons = neurons_age_df['neuron'].to_list()
        sorted_neurons = [neuron for neuron in sorted_neurons if neuron in heatmap_neurons]
        print ('neurons after adding from heatmap', sorted_neurons, len(sorted_neurons))
        corr_mat = corr_mat.reindex(sorted_neurons).reindex(sorted_neurons, axis=1)

        plt.title('Correlation Heatmap Ages sorted (' + type + ')')

    sns.heatmap(corr_mat, cmap='coolwarm', annot=False, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    plt.show()

def plot_violin(corr_matrix, nbr_df, pearson=True, type_filters=None):

    corr_str = 'Pearson Correlation' if pearson else 'Spearman Correlation'
    type_str = ""
    if type_filters:
        nbr_df = nbr_df[nbr_df['Type'].isin(type_filters)]
        type_str = "Electrical Junction, " if len(type_filters) == 1 else "Monadic Synapses, "

    correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  
                corr_value = corr_matrix[col1][col2]
                nbr_row = nbr_df[(nbr_df['Neuron 1'] == col1) & (nbr_df['Neuron 2'] == col2)]
                if nbr_row.empty:
                    nbr_row = nbr_df[(nbr_df['Neuron 1'] == col2) & (nbr_df['Neuron 2'] == col1)]
                
                nbr_value = 0 if nbr_row.empty else nbr_row['Nbr'].values[0]
                correlations.append((corr_value, nbr_value))

    corr_df = pd.DataFrame(correlations, columns=[corr_str, 'Nbr'])
    nbr_counts = corr_df['Nbr'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(x='Nbr', y=corr_str, data=corr_df, inner=None, color= None if pearson else 'orange') 
    title = f"{type_str}{'Pearson' if pearson else 'Spearman'} Correlation Distribution vs Nbr"
    plt.title(title)

    means = corr_df.groupby('Nbr')[corr_str].mean()
    unique_nbr_values = sorted(corr_df['Nbr'].unique())

    for i, nbr in enumerate(unique_nbr_values):
        count = nbr_counts.get(nbr, 0)
        plt.text(i, 1.05, f'n = {count}', horizontalalignment='center', size='small', color='black')
        group_data = corr_df[corr_df['Nbr'] == nbr][corr_str].dropna()
        rest_data = corr_df[corr_df['Nbr'] != nbr][corr_str].dropna()
        if len(group_data) > 0 and len(rest_data) > 0:
            stat, p_value = ranksums(group_data, rest_data)
            if p_value > 1e-2:
                formatted_p_value = f"{p_value:.2f}"  
            else:
                formatted_p_value = f"{p_value:.1e}"  
            plt.text(i, 1.15, f'p-value = {formatted_p_value}', horizontalalignment='center', size='small', color='black')
        mean_value = means[nbr]
        plt.plot([i - 0.2, i + 0.2], [mean_value, mean_value], color='darkblue', linewidth=2, linestyle='dashed')

    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=1.5, color='lightgray')  
    ax.grid(True, which='major', axis='x', linestyle='-', linewidth=1.5, color='lightgray') 
    ax.set_axisbelow(True)
    ax.set_yticks(np.arange(-1, 1.1, 0.25))
    plt.xlabel('Nbr')
    plt.ylabel(corr_str)
    plt.show()

def find_neruons_high_correlated_without_connectome_value(data, correlation_matrix, nbr_df, threshold=0.75):
    # List to store neuron pairs and their correlation
    neurons_nbr_0 = []

    # Loop through the correlation matrix and extract pairs with Nbr = 0
    for i, col1 in enumerate(correlation_matrix.columns):
        for j, col2 in enumerate(correlation_matrix.columns):
            if i < j:  # Only take one half of the matrix since it's symmetric
                # Get the corresponding Nbr value for the neuron pair
                nbr_row = nbr_df[(nbr_df['Neuron 1'] == col1) & (nbr_df['Neuron 2'] == col2)]
                nbr_row = nbr_df[(nbr_df['Neuron 1'] == col2) & (nbr_df['Neuron 2'] == col1)] if nbr_row.empty else nbr_row
                if nbr_row.empty:
                    corr_value = correlation_matrix[col1][col2]
                    if (corr_value > threshold):
                        neuron_data_1 = data[col1].dropna()
                        neuron_data_2 = data[col2].dropna()
                        aligned_data = pd.concat([neuron_data_1, neuron_data_2], axis=1).dropna()
                        if len(aligned_data) > 1:
                            corr, p_value = pearsonr(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                            neurons_nbr_0.append([col1, col2, corr, p_value])  # Store the neuron pair, correlation, and p-value


    # Create a DataFrame from the results
    neurons_nbr_0_df = pd.DataFrame(neurons_nbr_0, columns=['Neuron 1', 'Neuron 2', 'Pearson Correlation', 'P-Value'])

    return neurons_nbr_0_df

def main():
    combined_data = get_united_data_as_df()
    age_data = pd.read_csv('cengen_neuron_prediction.csv')
    pearson_corr = compute_correlation_matrix(combined_data, 'pearson')
    # plot_correlation_heapmap(pearson_corr, 'Pearson')
    spearman_corr = compute_correlation_matrix(combined_data, 'spearman')
    # plot_correlation_heapmap(spearman_corr, 'Spearman')
    nbr_df = pd.read_csv('NeuronConnect.csv')
    types = [['EJ'], ['S', 'R']]
    for filter in types:
        plot_violin(pearson_corr, nbr_df, type_filters=filter)
        plot_violin(spearman_corr, nbr_df, pearson=False, type_filters=filter)
    # neurons_nbr_0_df = find_neruons_high_correlated_without_connectome_value(combined_data, pearson_corr, nbr_df)
    # neurons_nbr_0_df.to_csv('Pearson_Nbr_0.csv', index=False)
    # neurons_nbr_0_df = find_neruons_high_correlated_without_connectome_value(combined_data, spearman_corr, nbr_df)
    # neurons_nbr_0_df.to_csv('Spearman_Nbr_0.csv', index=False)

if __name__ == '__main__':
    main()


