# Simple Python program for Diffie-Hellman Key Exchange

P = 23  # Prime number (public)
G = 9   # Primitive root (public)

# Alice chooses a private key
a = 4
# Bob chooses a private key
b = 6

# Calculate public keys
x = (G ** a) % P  # Alice's public key
y = (G ** b) % P  # Bob's public key

# Exchange public keys and generate shared secret
ka = (y ** a) % P  # Secret key for Alice
kb = (x ** b) % P  # Secret key for Bob

print("Publicly Shared Values: P =", P, " G =", G)
print("Alice's Private Key:", a)
print("Bob's Private Key:", b)
print("Alice's Public Key:", x)
print("Bob's Public Key:", y)
print("Secret Key for Alice:", ka)
print("Secret Key for Bob:", kb)
