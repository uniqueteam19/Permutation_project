import random
import math
import numpy as np

## 단계별 함수 설명##                                                # 단계별 계산 과정

                                                  
def generate_key(n, r,r2, r3):                                       
    """ 
    키 생성 함수 n,r을 사용하여 P(P(n,r) + 1, r2) 생성 중복된 순열값 배제.
    """ 
    P = math.perm(n,r) + 1                                          # P(n, r) + 1 계산
    selected_r3 = random.choice(r3)                                   # r3 배열에서 랜덤 값 선택 q
    selected_r2 = random.choice(r2)                                 # r2 배열에서 랜덤 값 선택 r2
    result_p_X_r2 = math.perm(P, selected_r2)                       # P(P, r2) 계산
    digits = [int(d) for d in str(result_p_X_r2) if d != '0']       # 각 자리수에서 0 제거
    S = [abs(d -selected_r3) for d in digits]                         # | 각 자리수 -q| 
    S = np.array(S)
    S = S.reshape(-1,1)                                             # S transposition
    dimension = len(S)                                 
    return S, dimension

def generate_random_matrix_and_inner_product(S, dim_S,q):
    """
    0 ~ q 까지 범위의 dim_S x dim_S matrix A 생성 및 A 내적 S 실행.
    """
    A = np.random.randint(0, q, size=(dim_S, dim_S))             # (dim_S) x (dim_S) 인 0 ~ q 범위의 A matrix 생성
    S_array = np.array(S)
    A_S = np.dot(A, S)                                              # A내적 S^T 계산
    return A, A_S

def number_to_binary_array(number): 
    """
    message를 binary로 변환.
    """
    binary_representation = format(number, 'b')                     # message m의 숫자를 비트로 변환
    binary_array = [int(bit) for bit in binary_representation]      # 변환한 비트를 리스트에 저장
    return binary_array

## b 계산 함수
def calculate_b_array(A_S, m, q, dim_S):
    """
    m의 길이만큼 A_S에 랜덤 e를 더하고 mod q 연산을 반복하여 b 배열 생성.
    """
    b_array = []
    for _ in m:
        n = dim_S
        e = np.random.randint(0, 2, size=(n, 1))                   # 작은 범위의 노이즈 벡터 e 생성 (0 ~ 1)
        b = np.mod(A_S + e, q)                                     # b 생성   b = (A 내적 S^T) + e (mod q) 
        b_array.append(b)  
    b_array = np.array(b_array)
    return b_array
def calculate_u_and_v(A, b_array, m, q, dim_S):
    """
    message 암호화 함수
    A^T와 b_array를 사용해 u와 v를 계산.
    u: A^T * X mod q.
    v: b^T * X + floor(q/2 * m) mod q.
    """
    U = []  
    V = []  
    X_array = []  

    A_T = A.T                                                       # A의 전치 행렬  
   
    for B, bit in zip(b_array, m):
        n = dim_S
        X = np.random.randint(0, 2, size=(n, 1))                    # 작은 범위의 랜덤 X 행렬 생성 (0 ~ 1)                                    
        X_array.append(X)
        u = np.dot(A_T, X)                                          # u 계산: A^T 내적 X (mod q)
        u = u % q
        U.append(u)  
        B_T = np.array(B).reshape(1, -1)
        inner_product = np.dot(B_T, X).item()                       # B^T * X 계산 (스칼라 값 추출)
        m_term = int(bit * (q // 2))                                # m * (q / 2)의 바닥 함수
        v = (inner_product + m_term) % q                            # v 계산: (B^T 내적 X) + floor(q/2 * m) (mod q)
        V.append(v) 
    U = np.array(U)
    return U, V, X_array  

def decrypt_v_to_d(v_array, S, U, q):
    """
    message 복호화 함수 
    d 계산 d = v - S^T 내적 u (mod q)
    d <= q/4 이면 0, d > q/4 이면 1로 변환 
    """
    d_array = [] 
    S_T = np.array(S).reshape(1, -1)                                # S를 전치 (1 x n 형태)
    for v, u in zip(v_array, U):
        inner_product = np.dot(S_T, u).item()                       # S^T 내적 u 
        d = np.mod(v - inner_product,q)                             # d 계산 d = v - S^T 내적 u (mod q)
        decoded_bit = 0 if d <= q // 4 else 1                       # d <= q/4 이면 0, d > q/4 이면 1로 변환 
        d_array.append(decoded_bit)

    return d_array

def bits_to_integer(d):
    """
    바이너리 리스트를 정수로 변환하는 함수.

    Parameters:
    bits (list of int): 0과 1로 이루어진 바이너리 리스트.

    Returns:
    int: 변환된 정수.
    """
    bit_str = ''.join(str(bit) for bit in d)                        # 바이너리 리스트를 문자열로 변환
    return int(bit_str, 2)                                          # 2진수 문자열을 정수로 변환

# 메인 실행 부분

n, r1 = 4, 2                                                        # n, r1 설정
r2 = [2,3,4]                                                        # r2 설정
r3 = [4,5,6,7,8,9]                                                  # r3 설정
S, dim_S = generate_key(n, r1, r2, r3)                              # 키 S 생성 함수
q = 2**20  
A, A_S = generate_random_matrix_and_inner_product(S, dim_S, q)      # A 생성 및 A 내적 S 생성, dim_S 계산
M = 165                                                             # message 생성
m = number_to_binary_array(M)                                       # message 바이너리로 변환 배열
b = calculate_b_array(A_S, m, q, dim_S)                             # b값 계산   b = (A 내적 S^T) + e (mod q) 
U, V, X_array = calculate_u_and_v(A, b, m, q, dim_S)                # 암호화 U, V 계산 u: A^T 내적 X (mod q)  v: (B^T 내적 X) + floor(q/2 * m) (mod q)
d = decrypt_v_to_d(V, S, U, q)                                      # 복호화 d = v - S^T 내적 u (mod q),  d <= q/4 이면 0, d > q/4 이면 1로 변환 
M1 = bits_to_integer(d)                                             # 바이너리 함수 문자열 변환 함수
print("메시지 M :", M)
print("암호화 전(Binary): ",m)
print("복호화 후(Binary): ",d)
print("메시지 복구 :",M1)

 # 사용자의 임의대로 설정하여 실행 #
