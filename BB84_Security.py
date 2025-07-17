from Crypto.Random import get_random_bytes
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def generate_bits(length):
    return [random.randint(0, 1) for _ in range(length)]

def generate_bases(length):
    return [random.choice(['+', 'x']) for _ in range(length)]

def encode_qubits(bits, bases):
    encoded_qubits = list(zip(bits, bases))
    return encoded_qubits

def if_eavesdropper(encoded_qubits, fraction=0):
    if fraction == 0:
        return encoded_qubits
    length = len(encoded_qubits) # Calculate the number of attacked qubits
    n_eavesdrop = int(length * fraction) # Randomly select the index of the attacked qubit
    eavesdrop_indices = set(random.sample(range(length), n_eavesdrop))
    eavesdrop_bases = generate_bases(length)
    attacked_qubits = [] # Create a new list of qubits
    for idx, (bit, encoding_basis) in enumerate(encoded_qubits):
        if idx in eavesdrop_indices:
            eavesdrop_basis = eavesdrop_bases[idx]
            if encoding_basis == eavesdrop_basis:
                eve_bit = bit
            else:
                eve_bit = random.randint(0, 1)
            attacked_qubits.append((eve_bit, eavesdrop_basis))
        else:
            attacked_qubits.append((bit, encoding_basis))
    return attacked_qubits

#def if_eavesdropper(encoded_qubits, fraction=0):
    if fraction == 0:
        return encoded_qubits   # no change
    length = len(encoded_qubits)
    n_eavesdrop = int(length * fraction)
    eavesdrop_indices = set(random.sample(range(length), n_eavesdrop))
    eavesdrop_bases = generate_bases(length)
    eavesdrop_bits = []
    for idx, ((bit, encoding_basis), eavesdrop_basis) in enumerate(zip(encoded_qubits, eavesdrop_bases)):
        if idx in eavesdrop_indices:
            if encoding_basis == eavesdrop_basis:
                eavesdrop_bits.append(bit)
            else:
                eavesdrop_bits.append(random.randint(0, 1))
        else:
            eavesdrop_bits.append(bit)
    eavesdrop_qubits = list(zip(eavesdrop_bits, eavesdrop_bases))
    return eavesdrop_qubits

#def if_eavesdropper(encoded_qubits, fraction=0):
    length = len(encoded_qubits)
    n_eavesdrop = int(length * fraction)
    # Randomly select the index to eavesdrop on
    eavesdrop_indices = set(random.sample(range(length), n_eavesdrop))
    eavesdrop_bases = generate_bases(length)
    eavesdrop_bits = []
    for idx, ((bit, encoding_basis), eavesdrop_basis) in enumerate(zip(encoded_qubits, eavesdrop_bases)):
        if idx in eavesdrop_indices:
            if encoding_basis == eavesdrop_basis:
                eavesdrop_bits.append(bit)
            else:
                eavesdrop_bits.append(random.randint(0, 1))
        else:
            eavesdrop_bits.append(bit)
    eavesdrop_qubits = list(zip(eavesdrop_bits, eavesdrop_bases))
    encoded_qubits = eavesdrop_qubits
    return encoded_qubits

def measure_qubits(encoded_qubits, measuring_bases):
    measured_bits = []
    for (bit, encoding_basis), measuring_basis in zip(encoded_qubits, measuring_bases):
        if encoding_basis == measuring_basis:
            measured_bits.append(bit)
        else:
            measured_bits.append(random.randint(0, 1))
    return measured_bits

def sift_key(sender_bases, receiver_bases, bits):
    sifted_key = [bit for s_base, r_base, bit in zip(sender_bases, receiver_bases, bits) if s_base == r_base]
    return sifted_key

def sift_accuracy(s_bit, r_bit):
    if not (s_bit and r_bit) or len(s_bit) != len(r_bit):
        return 0.0
    correct = sum(1 for s, r in zip(s_bit, r_bit) if s == r)
    return correct / len(s_bit) if s_bit else 0

#**************************************************

if __name__ == '__main__':
    trials = 1000
    bit_lengths = [100, 500, 1000]  # three kinds of bit length
    eavesdrop_accuracies_dict = {}
    eavesdrop_acc_mean = []

    for length in bit_lengths:
        eavesdrop_accuracies = []
        for _ in range(trials):
            #process of Alice and Bob
            alice_bits = generate_bits(length)
            alice_bases = generate_bases(length)
            bob_bases = generate_bases(length)

            # Eve attack
            encoded_qubits = encode_qubits(alice_bits, alice_bases)
            encoded_qubits = if_eavesdropper(encoded_qubits,fraction=1) # eavesdropping
            bob_results_eve = measure_qubits(encoded_qubits, bob_bases)
            alice_key_eve = sift_key(alice_bases, bob_bases, alice_bits)
            bob_key_eve = sift_key(alice_bases, bob_bases, bob_results_eve)
            acc_eve = sift_accuracy(alice_key_eve, bob_key_eve)
            eavesdrop_accuracies.append(acc_eve)

        # store accuracy of each length
        eavesdrop_accuracies_dict[length] = eavesdrop_accuracies
        #statistics data
        accuracies_np = np.array(eavesdrop_accuracies)
        mean = np.mean(accuracies_np)
        eavesdrop_acc_mean.append(mean)
        median = np.median(accuracies_np)
        std = np.std(accuracies_np)
        q25 = np.percentile(accuracies_np, 25)
        q75 = np.percentile(accuracies_np, 75)

        print(f'\nBit length: {length}')
        print(f'Sift Key length: {len(alice_key_eve)}')
        print(f'Mean accuracy:       {mean:.4f}')
        print(f'Median accuracy:     {median:.4f}')
        print(f'Standard deviation:  {std:.4f}')
        print(f'IQR (middle 50%):    {q25:.4f} to {q75:.4f}')
        mu = mean      # average value
        sigma = std  # standard deviation
        z = (0.9 - mu) / sigma
        p_exceed = 1 - norm.cdf(z)
        print(f'Prob(X ≥ 0.9) ≈ {p_exceed:.3f}')
    # draw a combined graph
    x = list(range(1, trials + 1))
    for length in bit_lengths:
        plt.plot(x, eavesdrop_accuracies_dict[length], label=f'{length} bits')
    plt.xlabel('Trial')
    plt.ylabel('Sift Key Accuracy with Eavesdropping')
    plt.title('BB84: Eavesdropping Accuracy at Different Bit Lengths')
    plt.ylim(0.5, 1.0)
    plt.yticks(np.arange(0.5, 1.01, 0.1))
    plt.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
    plt.legend()
    # draw seperated graph
    colors = ['C0', 'C1', 'C2']
    for length in bit_lengths:
        plt.figure()  # creat new graphs
        plt.plot(x, eavesdrop_accuracies_dict[length], label=f'{length} bits', color=colors[bit_lengths.index(length)])
        plt.xlabel('Trial')
        plt.ylabel('Sift Key Accuracy with Eavesdropping')
        plt.title(f'BB84: Eavesdropping Accuracy ({length} bits)')
        plt.ylim(0.5, 1.0)
        yticks = list(np.arange(0.5, 1.01, 0.1))
        mean_tick = round(eavesdrop_acc_mean[bit_lengths.index(length)], 2)
        if not np.any(np.isclose(yticks, mean_tick, atol=1e-6)):
            yticks.append(mean_tick)
            yticks = sorted(yticks)
        plt.yticks(yticks)
        plt.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
        plt.axhline(y=eavesdrop_acc_mean[bit_lengths.index(length)], color=colors[bit_lengths.index(length)], linestyle='-.', linewidth=1)
        plt.legend()
    plt.show()