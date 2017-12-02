def perceptron(x1, x2, w1, w2, theta):
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


def AND(x1, x2):
    return perceptron(x1, x2, 0.5, 0.5, 0.7)


def OR(x1, x2):
    return perceptron(x1, x2, 0.5, 0.5, 0.3)


def NAND(x1, x2):
    return perceptron(x1, x2, -0.5, -0.5, -0.7)


print("AND")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print("OR")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))

# NAND
print("NAND")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))
