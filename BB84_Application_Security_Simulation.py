import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def generate_bits(length):
    """Generate a random bit string of the given length."""
    return [random.randint(0, 1) for _ in range(length)]

def generate_bases(length):
    """Generate a random list of bases ('+' or 'x') for quantum encoding."""
    return [random.choice(['+', 'x']) for _ in range(length)]

def encode_qubits(bits, bases):
    """Pair each bit with its corresponding encoding basis."""
    return list(zip(bits, bases))

def QBER_Contribution(encoded_qubits, qber):
    """
    Introduce quantum bit errors according to the specified QBER (Quantum Bit Error Rate).
    A random subset of qubits are flipped to simulate the effect of quantum noise.
    """
    length = len(encoded_qubits)
    n_errors = int(length * qber)
    error_indices = set(random.sample(range(length), n_errors))
    noisy_qubits = encoded_qubits.copy()
    for idx in error_indices:
        bit, basis = noisy_qubits[idx]
        noisy_qubits[idx] = (1 - bit, basis)  # Bit flip to simulate noise
    return noisy_qubits

def measure_qubits(encoded_qubits, measuring_bases):
    """
    Measure each qubit according to the specified basis.
    If the basis matches the encoding basis, the measurement yields the original bit.
    Otherwise, the result is a random bit (simulating quantum measurement uncertainty).
    """
    measured_bits = []
    for (bit, encoding_basis), measuring_basis in zip(encoded_qubits, measuring_bases):
        if encoding_basis == measuring_basis:
            measured_bits.append(bit)
        else:
            measured_bits.append(random.randint(0, 1))
    return measured_bits

def sift_key(sender_bases, receiver_bases, bits):
    """
    Sift the key by keeping only those bits where sender's and receiver's bases match.
    This is the standard BB84 sifting procedure.
    """
    return [bit for s_base, r_base, bit in zip(sender_bases, receiver_bases, bits) if s_base == r_base]

def sift_accuracy(s_bit, r_bit):
    """
    Calculate the accuracy of the sifted key: proportion of bits that match between Alice and Bob.
    """
    if not s_bit or not r_bit or len(s_bit) != len(r_bit):
        return 0.0
    correct = sum(1 for s, r in zip(s_bit, r_bit) if s == r)
    return correct / len(s_bit) if s_bit else 0

#************ Main Experiment Code ************
if __name__ == '__main__':
    trials = 1000
    bit_lengths = [500]  # You can change to [500, 1000, 2000] for more curves
    contribution_min = 0.055
    contribution_max = 0.195

    eavesdrop_accuracies_dict = {}

    for length in bit_lengths:
        accuracies = []
        for _ in range(trials):
            # For each trial, QBER is sampled randomly from [min, max]
            qber = random.uniform(contribution_min, contribution_max)
            alice_bits = generate_bits(length)
            alice_bases = generate_bases(length)
            bob_bases = generate_bases(length)

            encoded_qubits = encode_qubits(alice_bits, alice_bases)
            noisy_qubits = QBER_Contribution(encoded_qubits, qber)
            bob_results = measure_qubits(noisy_qubits, bob_bases)

            alice_key = sift_key(alice_bases, bob_bases, alice_bits)
            bob_key = sift_key(alice_bases, bob_bases, bob_results)
            acc = sift_accuracy(alice_key, bob_key)
            accuracies.append(acc)

        eavesdrop_accuracies_dict[length] = accuracies
        # Statistical analysis of sifted key accuracy
        accuracies_np = np.array(accuracies)
        mean = np.mean(accuracies_np)
        median = np.median(accuracies_np)
        std = np.std(accuracies_np)
        q25 = np.percentile(accuracies_np, 25)
        q75 = np.percentile(accuracies_np, 75)
        print(f'\nBit length: {length}')
        print(f'Mean accuracy:       {mean:.4f}')
        print(f'Median accuracy:     {median:.4f}')
        print(f'Standard deviation:  {std:.4f}')
        print(f'IQR (middle 50%):    {q25:.4f} to {q75:.4f}')
        mu = mean
        sigma = std
        z = (0.9 - mu) / sigma
        p_exceed = 1 - norm.cdf(z)
        print(f'Prob(X ≥ 0.9) ≈ {p_exceed:.3f}')

    # Plotting the results (multiple curves if using multiple bit lengths)
eavesdrop_acc_mean = []
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']  

x = list(range(1, trials + 1))
for idx, length in enumerate(bit_lengths):
    accuracies = eavesdrop_accuracies_dict[length]
    mean_tick = round(np.mean(accuracies), 2)
    eavesdrop_acc_mean.append(mean_tick)

    plt.figure()
    plt.plot(x, accuracies, label=f'{length} bits', color=colors[idx])
    plt.xlabel('Trial')
    plt.ylabel('Sift Key Accuracy (QBER noise)')
    plt.title(f'BB84: Sift Key Accuracy ({length} bits)')
    plt.ylim(0.5, 1.0)
    yticks = list(np.arange(0.5, 1.01, 0.1))
    if not np.any(np.isclose(yticks, mean_tick, atol=1e-6)):
        yticks.append(mean_tick)
        yticks = sorted(yticks)
    plt.yticks(yticks)
    plt.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='Accuracy = 0.9')
    plt.axhline(y=mean_tick, color=colors[idx], linestyle='-.', linewidth=1, label=f'Mean = {mean_tick:.2f}')
    plt.legend()

plt.show()