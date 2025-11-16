# Python program to implement Hill Cipher

# Initialize matrices
keyMatrix = [[0] * 3 for i in range(3)]
messageVector = [[0] for i in range(3)]
cipherMatrix = [[0] for i in range(3)]

# Function to generate key matrix from key string
def getKeyMatrix(key):
    k = 0
    for i in range(3):
        for j in range(3):
            keyMatrix[i][j] = ord(key[k]) % 65
            k += 1

# Function to encrypt the message
def encrypt(messageVector):
    for i in range(3):
        cipherMatrix[i][0] = 0
        for x in range(3):
            cipherMatrix[i][0] += keyMatrix[i][x] * messageVector[x][0]
        cipherMatrix[i][0] = cipherMatrix[i][0] % 26

def HillCipher(message, key):
    # Generate key matrix
    getKeyMatrix(key)

    # Generate message vector
    for i in range(3):
        messageVector[i][0] = ord(message[i]) % 65

    # Encrypt the message
    encrypt(messageVector)

    # Generate ciphertext
    CipherText = []
    for i in range(3):
        CipherText.append(chr(cipherMatrix[i][0] + 65))

    print("Ciphertext:", "".join(CipherText))

# Driver code
def main():
    message = "ACT"
    key = "GYBNQKURP"
    HillCipher(message, key)

if __name__ == "__main__":
    main()
