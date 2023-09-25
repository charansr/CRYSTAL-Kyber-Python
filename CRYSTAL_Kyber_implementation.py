import random
import numpy as np

# Random Coefficient Generator
def gen_random_coeffs(min_value = -10, max_value = 10, list_length = 4):
    random_integers = [random.randint(min_value, max_value) for _ in range(list_length)]
    return random_integers

# Polynomial Matrix Multiplier
def matrix_multiply(A, B):
    # Create an empty result matrix
    result = np.zeros((A.shape[0], B.shape[1]),dtype=object)
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i,j]=np.poly1d([1])
    # Perform matrix multiplication
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            for k in range(A.shape[1]):
                result[i, j] += np.polymul(A[i, k],B[k, j])
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            result[i,j] = np.polydiv(result[i,j],modulo_poly_f)[1]
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            iv=0
            for v in result[i,j]:
                result[i,j][iv] = (result[i,j][iv] % modulo_val_q)
                iv+=1
    return result

# Polynomial Matrix Addition
def add_matrices(matrix1, matrix2):
    #if matrix1.shape != matrix2.shape:
    #    raise ValueError("Matrices must have the same dimensions for addition.")

    rows, cols,poly = matrix1.shape
    result = np.empty((rows, cols),dtype=object)
    for i in range(rows):
            for j in range(cols):
                result[i,j]=np.poly1d([1])

    for i in range(rows):
        for j in range(cols):
            result[i, j] = np.polyadd(matrix1[i, j],matrix2[i, j])

    return result

# Takes Modulo of Polynomial
def poly_mod_func(A):
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            iv=0
            for v in A[i,j]:
                A[i,j][iv] = (A[i,j][iv] % modulo_val_q)
                iv+=1
    return A

# Rounds value to modulo constants
def closest_to_number(number):
    number=abs(number)
    # Calculate the absolute differences
    diff_to_0 = abs(number - 0)
    diff_to_17 = abs(number - modulo_val_q)
    diff_to_9 = abs(number - modulo_multiplier)

    # Compare the differences and return the result
    if diff_to_0 < diff_to_17 and diff_to_0 < diff_to_9:
        return 0
    elif diff_to_17 < diff_to_0 and diff_to_17 < diff_to_9:
        return modulo_multiplier
    else:
        return modulo_multiplier
# Message vector
original_message=[]
print("Input a binary value: ")
for i in range(4):
    val = int(input("Type a 1 or 0: "))
    original_message+=[val]
message_vector=np.array([[np.poly1d(original_message)]])

# Modulo Constants
modulo_val_q = 17
modulo_poly_f = np.poly1d([1,0,0,0,1])

# Private Key
private_key_s = np.array([[np.poly1d([1]+gen_random_coeffs(min_value=-1,max_value=1,list_length=2)),np.poly1d([1]+gen_random_coeffs(min_value=-1,max_value=1,list_length=2))]])

# Public Keys
public_key_A = np.array([[np.poly1d([1]+gen_random_coeffs(list_length=3)),np.poly1d([1]+gen_random_coeffs(list_length=3))],[np.poly1d([1]+gen_random_coeffs(list_length=3)),np.poly1d([1]+gen_random_coeffs(list_length=3))]])

# Error Vector for calculation of second public key(t)
error_vector = np.array([[np.poly1d([0.0,1,0,0]),np.poly1d([0.0,1,-1,0])]])


# Calculating public key t
public_key_t=matrix_multiply(private_key_s,public_key_A)
public_key_t=np.array([[public_key_t[0,0],public_key_t[0,1]]])
public_key_t=add_matrices(public_key_t,error_vector)

#Encryption Steps

randomizer_vector=np.array([[np.poly1d([-1,1,0,0]),np.poly1d([1,1,0,-1])]])
error_vector_1 = np.array([[np.poly1d([0,1,1,0]),np.poly1d([0,1,0,0])]])
error_vector_2 = np.array([[np.poly1d([-1,-1,0,0])]])


modulo_multiplier=int(((float)(modulo_val_q)/2.0)+0.5)

message_vector[0][0]=np.polymul(message_vector[0][0],modulo_multiplier)

ciphertext_u=add_matrices(error_vector_1,matrix_multiply(randomizer_vector,np.transpose(public_key_A,(1, 0, 2))))

temporary_public_key_t=np.zeros((1,2,4))
iloc=0
for i in public_key_t:
    jloc=0
    for j in i:
        kloc=0
        for k in j:
            temporary_public_key_t[iloc][jloc][kloc]=public_key_t[iloc][jloc][kloc]
            kloc+=1
        jloc+=1
    iloc+=1

ciphertext_v=add_matrices(message_vector,add_matrices(error_vector_2,matrix_multiply(randomizer_vector,np.transpose(temporary_public_key_t,(1,0,2)))))
ciphertext_v=poly_mod_func(ciphertext_v)

# Decrypt Message
noisy_message = ciphertext_v - matrix_multiply(ciphertext_u,np.transpose(private_key_s,(1,0,2)))


final_message=""
li=0
for i in noisy_message[0][0]:
    noisy_message[0][0][li]=closest_to_number(noisy_message[0][0][li])/9
    li+=1
for i in noisy_message[0][0]:
    final_message+=str(int(i))+" "
print(final_message)



