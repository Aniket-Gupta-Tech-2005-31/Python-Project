# Python program to implement MD5 Algorithm
import hashlib

# Create MD5 hash objects
result = hashlib.md5(b'Ismile')
result1 = hashlib.md5(b'Esmile')

# Print the byte equivalent of the MD5 hashes
print("The byte equivalent of MD5 hash for 'Ismile' is:", result.digest())
print("The byte equivalent of MD5 hash for 'Esmile' is:", result1.digest())

# Print the hexadecimal equivalent (easier to read)
print("The hexadecimal equivalent of MD5 hash for 'Ismile' is:", result.hexdigest())
print("The hexadecimal equivalent of MD5 hash for 'Esmile' is:", result1.hexdigest())
