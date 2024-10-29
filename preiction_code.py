from autoencoder_estimator import *
from sample_parser import *
from autoencoder_model import * 
from neuron_clustering import get_clustering
from scipy import stats
from scipy.stats import wilcoxon

def test_mask(samples_series, test_sample_index):
    autoencoder_estimator = AutoEncoderAcitivityPredictionEstimator()
    
    all_names_np = get_all_names_np(samples_series)
    test_sample = samples_series[test_sample_index]

    train_samples = samples_series[:test_sample_index] + samples_series[test_sample_index+1:]
    activity, _, _ = build_input_output_table(train_samples, all_names_np, smooth_data=True)

    autoencoder = autoencoder_model.ImprovedMissingDataAutoencoder(
        activity,
        encoding_dim=autoencoder_estimator.encoding_dim,
        dropout_rate=autoencoder_estimator.dropout_rate,
        masking_prob=autoencoder_estimator.masking_prob
    )
    autoencoder.train(epochs=autoencoder_estimator.epochs)

    test_activity, _, _ = build_input_output_table([test_sample], all_names_np, smooth_data=True)

    return autoencoder_estimator, test_activity, autoencoder, all_names_np


def calc_reconstruct_error(test_activity, all_names_np, neurons_to_mask, autoencoder_estimator, autoencoder):
    return autoencoder_estimator._mask_and_calc_reconstruct_error(test_activity, all_names_np, neurons_to_mask, autoencoder)

def select_neurons_to_mask(all_names_np, percentage_to_mask, test_activity, clusters_list=[]):
    if (percentage_to_mask == 0):
        return []
    available_neurons_mask = ~np.isnan(test_activity[0,:])
    if (np.sum(available_neurons_mask) < 10):
        print ("@@@ Less than 10 neurons available to mask")
    num_neurons_to_mask = int(np.sum(available_neurons_mask) * (percentage_to_mask / 100))
    if (clusters_list==[]):
        neurons_to_mask = random.sample(list(all_names_np[available_neurons_mask]), num_neurons_to_mask)
    else:
        neurons_to_mask = []
        available_neurons_set = set(all_names_np[available_neurons_mask])  
        
        for cluster in clusters_list:
            available_cluster_neurons = [neuron for neuron in cluster if neuron in available_neurons_set]
            if available_cluster_neurons:
                num_neurons_in_cluster_to_mask = int(len(available_cluster_neurons) * (percentage_to_mask / 100))
                num_neurons_in_cluster_to_mask = max(1, num_neurons_in_cluster_to_mask)  # Ensure at least one neuron is masked
                neurons_to_mask.extend(random.sample(available_cluster_neurons, num_neurons_in_cluster_to_mask))
    
    return neurons_to_mask

def check_statistical_difference(random_errors_dict, clustered_errors_dict):
    """
    Perform a paired t-test or Wilcoxon signed-rank test to check for a significant difference
    between random and clustered masking over all percentages.
    """
    # Flatten the lists of errors for all percentages into single lists
    random_errors_flat = []
    clustered_errors_flat = []
    
    for percentage in random_errors_dict:
        random_errors_flat.extend(random_errors_dict[percentage])
        clustered_errors_flat.extend(clustered_errors_dict[percentage])
    
    # Convert to numpy arrays
    random_errors_flat = np.array(random_errors_flat)
    clustered_errors_flat = np.array(clustered_errors_flat)

    # Paired t-test
    t_statistic, p_value_ttest = stats.ttest_rel(random_errors_flat, clustered_errors_flat)
    print(f" - Paired t-test p-value: {p_value_ttest:.5f}")

    # Wilcoxon signed-rank test (use if normality assumption does not hold)
    differences = random_errors_flat - clustered_errors_flat
    if np.all(differences == 0):
        print("All differences are zero, skipping Wilcoxon test.")
    else:
        wilcoxon_statistic, p_value_wilcoxon = wilcoxon(random_errors_flat, clustered_errors_flat)

        # Print results
        print(f" - Wilcoxon test p-value: {p_value_wilcoxon:.5f}")
    
    return p_value_ttest, p_value_wilcoxon


def main(rounds):
    samples_series = AllSampleSeries(atlas=True, zimmer=False, flavell=False).get_all_samples(filter_out_heat=False, require_annotation=True)

    clustered_neurons = get_clustering()

    percentages = list(range(0, 100, 10))
    random_samples_indexes = random.sample(range(len(samples_series)), rounds)
    random_errors_dict = {}
    clustered_errors_dict = {}

    i=1
    for sample_index in random_samples_indexes:
        print('round ', i)
        autoencoder_estimator, test_activity, autoencoder, all_names_np = test_mask(samples_series, sample_index)
        i+=1
        for percentage in percentages:
            random_neurons_to_mask = select_neurons_to_mask(all_names_np, percentage, test_activity)
            clustered_neurons_to_mask = select_neurons_to_mask(all_names_np, percentage, test_activity, clustered_neurons)
            random_error = calc_reconstruct_error(test_activity, all_names_np, random_neurons_to_mask, autoencoder_estimator, autoencoder)
            clustered_error = calc_reconstruct_error(test_activity, all_names_np, clustered_neurons_to_mask, autoencoder_estimator, autoencoder)
            clustered_errors_dict[percentage] = clustered_errors_dict.get(percentage, []) + [clustered_error]
            random_errors_dict[percentage] = random_errors_dict.get(percentage, []) + [random_error]
    
    random_errors = []
    clustered_errors = []
    for percentage in percentages:
        random_errors.append(np.mean(random_errors_dict[percentage]))
        clustered_errors.append(np.mean(clustered_errors_dict[percentage]))

    p_value_ttest, p_value_wilcoxon = check_statistical_difference(random_errors_dict, clustered_errors_dict)

    plt.figure(figsize=(10, 6))
    plt.plot(percentages, random_errors, marker='o', label='Random Masking', color='lightsalmon')
    plt.plot(percentages, clustered_errors, marker='s', label='Clustered Masking', color='blue')
    plt.title('Reconstruction Error vs Percentage of Masked Neurons, {} rounds'.format(rounds))
    plt.xlabel('Percentage of Masked Neurons')
    plt.ylabel('Reconstruction Error')
    plt.xticks(np.arange(0, 100, 10))
    plt.gca().invert_xaxis()
    plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5) 
    plt.legend()
    plt.grid(True)
    if p_value_ttest is not None:
        plt.text(50, max(random_errors), f't-test p-value: {p_value_ttest:.5f}', color='black', fontsize=12)
    if p_value_wilcoxon is not None:
        plt.text(50, max(random_errors) - 0.05, f'Wilcoxon p-value: {p_value_wilcoxon:.5f}', color='black', fontsize=12)
    # plt.show()

   
main(12)
main(9)
main(6)
plt.show()
