import random
import math
#輸入:二  隱藏層:一層兩顆  輸出:一

weight = 0.9 #初始化權重
w11 = random.uniform(-weight, weight)
w21 = random.uniform(-weight, weight)
b1 = 0

w12 = random.uniform(- weight,weight)
w22 = random.uniform(-weight,weight)
b2 = 0

o1 = random.uniform(-weight,weight)
o2 = random.uniform(-weight,weight)
ob = 0

def sigmoid(x): #sigmoid 函數
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_prime(x): #sigmoid 的導數
    return x * (1 - x)

def trainning(i1,i2,target,learning_rate):
    global w11,w21,b1,w12,w22,b2
    global o1,o2,ob
    
    s1 = w11 * i1 + w21 * i2 + b1
    s1 = sigmoid(s1)
    s2 = w12 * i1 + w22 * i2 + b2
    s2 = sigmoid(s2)
    
    output = s1 * o1 + s2 * o2 + ob
    output = sigmoid(output)
    
    bias = target - output
    derror_g = bias * sigmoid_prime(output)
    
    ds1 = derror_g* o1 * sigmoid_prime(s1)
    ds2 = derror_g * o2 * sigmoid_prime(s2)

    o1 += learning_rate * s1 * derror_g
    o2 += learning_rate* s2 * derror_g
    ob += learning_rate * derror_g
    
    w11 += learning_rate * i1 * ds1
    w21 += learning_rate * i2 * ds1
    b1 += learning_rate * ds1
    w12 += learning_rate * i1 * ds2
    w22 += learning_rate * i2 * ds2
    b2 += learning_rate * ds2

INPUTS = [[0,0],[0,1],[1,0],[1,1]]
OUTPUTS = [[0],[1],[1],[0]]

def predict(i1,i2):    
    s1 = w11 * i1 + w21 * i2 + b1
    s1 = sigmoid(s1)
    s2 = w12 * i1 + w22 * i2 + b2
    s2 = sigmoid(s2)
    
    output = s1 * o1 + s2 * o2  + ob
    output = sigmoid(output)
    
    return output


for epoch in range(1,10001):
    indexes = [0,1,2,3]
    random.shuffle(indexes)
    for j in indexes:
        trainning(INPUTS[j][0],INPUTS[j][1],OUTPUTS[j][0], learning_rate=0.2)
    
    if epoch%1000 == 0:
        cost = 0
        for j in range(4):
            o = predict(INPUTS[j][0],INPUTS[j][1])
            cost += (OUTPUTS[j][0] - o) ** 2
        cost /= 4
        print("訓練", epoch, "次後的平均平方誤差為:", cost)       
        

for i in range(4):
    result = predict(INPUTS[i][0],INPUTS[i][1])
    print("對於輸入", INPUTS[i], "預期", OUTPUTS[i][0], "預測大約為", f"{result:4.4}","這是個"+"正確的預測" if round(result)==OUTPUTS[i][0] else "不正確預測")
    #print("隱藏層最終權重:",w11,w12,w21,w22)
    #print("輸出層最終權重:",o1,o2)
