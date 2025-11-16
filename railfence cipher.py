def RailFence(txt):
    result = ""
    for i in range(len(txt)):
        if i % 2 == 0:
            result += txt[i]

    for i in range(len(txt)):
        if i % 2 != 0:
            result += txt[i]
    return result

string = input("Enter a string: ")
print("Encrypted:", RailFence(string))
