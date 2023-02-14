def trainning():
    #初始化權重、臨界點和學習速度
    weight = [0.4,0.4]
    threshold = 0.2
    learning_rate = 0.1

    #訓練數據
    trainning_A = [[0,0],[0,1],[1,0],[1,1]]
    trainning_B = [0,0,0,1]

    bias = 1 #總誤差
    y = 0 #實際输出
    loop = 1 #訓練次數
    while bias != 0:
        print("第",loop,"次訓練")
        bias = 0
        e = 0 #分別誤差
        i = 0 #總誤差
        for x in trainning_A:
            X = x[0] * weight[0] + x[1] * weight[1] - threshold
            y = stepFuncion(X)
            #誤差e = 期望输出 - 實際输出
            e = trainning_B[i] - y

            if e != 0:
                # 調整權重
                weight[0] += learning_rate * x[0] * e
                weight[1] += learning_rate * x[1] * e
                bias += abs(e)
            i += 1
            print("x:", x, " y:", y, " 誤差：",e," 修改過後權重為：w1:",round(weight[0],3)," w2:",round(weight[1],3))
	    #權重有取小數點輸出
        loop += 1

    print("訓練完成")
    return weight,threshold  
def stepFuncion(x):  
    if x > 0:
        return 1
    else:
        return 0

def predict(test_A):
    print("測試數據：",test_A)
    print("預測如下：")
    for x in test_A:
        X = x[0] * weight[0] + x[1] * weight[1] - threshold
        y = stepFuncion(X)
        print("x:", x, " y:", y)

weight,threshold = trainning()
test_A = [[1,0],[0,1],[1,1],[0,0]]  #測試數據
predict(test_A)
