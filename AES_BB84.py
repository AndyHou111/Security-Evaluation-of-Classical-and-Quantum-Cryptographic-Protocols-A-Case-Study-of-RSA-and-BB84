# AES Encryption and Decryption example using PyCryptodome
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# AES encryption function
def aes_encrypt(plaintext, key):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode('utf-8'))
    return nonce, ciphertext, tag

# AES decryption function
def aes_decrypt(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    try:
        cipher.verify(tag)
        return plaintext.decode('utf-8')
    except ValueError:
        return "Decryption failed!"

# BB84 quantum key distribution simulation
import random
# generating a list of randomly choosing bits 0 and 1 with certain length
def generate_bits(length):
    return [random.randint(0, 1) for _ in range(length)]
# generating a list of randomly choosing bases + and x with certain length
def generate_bases(length):
    return [random.choice(['+', 'x']) for _ in range(length)]
# generating the encode_qubits
def encode_qubits(bits, bases):
    encoded_qubits = list(zip(bits, bases))
    return encoded_qubits

def if_eavesdropper(encoded_qubits):
    # Eavesdropper randomly chooses bases and measures the qubits
    eavesdrop_bases = generate_bases(len(encoded_qubits))
    eavesdrop_bits = []
    for (bit, encoding_basis), eavesdrop_basis in zip(encoded_qubits, eavesdrop_bases):
        if encoding_basis == eavesdrop_basis:
            eavesdrop_bits.append(bit)
        else:
            eavesdrop_bits.append(random.randint(0, 1))
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

# Demonstration
if __name__ == '__main__':
    # AES demonstration
    aes_key = get_random_bytes(16)  # AES-128 key
    plaintext = 'Hello, World! '
    temptext = plaintext
    for i in range(20):
        plaintext += temptext
    nonce, ciphertext, tag = aes_encrypt(plaintext, aes_key)
    decrypted_text = aes_decrypt(nonce, ciphertext, tag, aes_key)

    print('AES Example:')
    print(f'Plaintext: {plaintext}')
    print(f'Ciphertext: {ciphertext.hex()}')
    print(f'Decrypted text: {decrypted_text}\n')

    # BB84 demonstration
    length = 100
    alice_bits = generate_bits(length)
    alice_bases = generate_bases(length)
    bob_bases = generate_bases(length)

    encoded_qubits = encode_qubits(alice_bits, alice_bases)
    bob_results = measure_qubits(encoded_qubits, bob_bases)

    alice_key = sift_key(alice_bases, bob_bases, alice_bits)
    bob_key = sift_key(alice_bases, bob_bases, bob_results)

    print('BB84 Quantum Key Distribution Simulation:')
    print(f'Alice sifted key: {alice_key}')
    print(f'Bob sifted key:   {bob_key}')
    print(f'Keys match: {alice_key == bob_key}')
    print(f'shift key accuracy: {sift_accuracy(alice_key, bob_key):.2%}')

    #BB84 Eavesdropper Simulation
    print('\nBB84 Eavesdropper Simulation:')
    encoded_qubits = if_eavesdropper(encoded_qubits)
    bob_results =  measure_qubits(encoded_qubits, bob_bases)
    alice_key = sift_key(alice_bases, bob_bases, alice_bits)
    bob_key = sift_key(alice_bases, bob_bases, bob_results)
    #true if all sift_key are matched. 
    print(f'Keys match: {alice_key == bob_key}')
    #sift_accuracy is the proportion of number of matched keys in the number of total sift keys. (percentage)
    print(f'shift key accuracy: {sift_accuracy(alice_key, bob_key):.2%}')

    