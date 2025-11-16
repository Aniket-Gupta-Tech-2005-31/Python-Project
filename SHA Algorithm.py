import hashlib

# Get input from the user
input_str = input("Enter the value to encode: ")

# Encode the input and generate SHA1 hash
result = hashlib.sha1(input_str.encode())

# Print the SHA1 hash in hexadecimal format
print("The hexadecimal equivalent of SHA1 is:")
print(result.hexdigest())
