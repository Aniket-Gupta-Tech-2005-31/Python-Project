# Simple Python program for Monoalphabetic Cipher

import string
import random

# Create alphabet and random key
alphabet = string.ascii_uppercase
key = ''.join(random.sample(alphabet, len(alphabet)))

def encrypt(plaintext):
    plaintext = plaintext.upper()
    ciphertext = ''
    for ch in plaintext:
        if ch.isalpha():
            ciphertext += key[alphabet.index(ch)]
        else:
            ciphertext += ch
    return ciphertext

def decrypt(ciphertext):
    plaintext = ''
    for ch in ciphertext:
        if ch.isalpha():
            plaintext += alphabet[key.index(ch)]
        else:
            plaintext += ch
    return plaintext

# Example
text = "HELLO"
print("Original Text:", text)
print("Key:", key)

encrypted = encrypt(text)
print("Encrypted Text:", encrypted)

decrypted = decrypt(encrypted)
print("Decrypted Text:", decrypted)
