import numpy as np
import random


population_size = 10 #種群數量
pc = 0.7             #交配率
pm = 0.001           #突變率
generations = 10     #代數
DNA_length = 4       #染色體長度
x_bound = [0, 15]    # 變量



def Fuction_fitness(x):
    return 15 * x - x ** 2

class GA(object):
    #產生隨機染色體
    def __init__(self):
        self.populations = np.random.randint(0, 2, (population_size, DNA_length))
 
    # DNA解碼(2進位轉10進位)
    def DNA_decode(self, DNA):
        return np.dot(DNA, 2 ** np.arange(DNA_length)) / (2 ** DNA_length-1) * x_bound[1]

    #calculatefitness
    def calculate_fitness(self, DNA):
        DNA_value = self.DNA_decode(DNA)#DNA解碼
        fitness = Fuction_fitness(DNA_value)#代入適合度公式
        if fitness.any() < 0:#去掉有可能的負數fitness
            fitness=fitness*-1
        return fitness

    def selection(self):
        DNA_id = np.random.choice(np.arange(population_size), size=2, replace=False)# 從0~DNA.length-1中隨機選擇2組DNA編號
        DNA = self.populations[DNA_id]
        DNA_fitness = self.calculate_fitness(DNA)#計算適合度
        return DNA[np.argsort(DNA_fitness)], DNA_id#排列個染色體適合度

    #交配
    def crossover(self, DNA):
        crossover_points = np.empty((DNA_length)).astype(bool)
        for i in range(DNA_length):
            if random.random() < pc: #小於交配率就交換
                crossover_points[i] = True
            else: 
                False
        DNA[0, crossover_points] = DNA[1, crossover_points]
        return DNA

    #突變   
    def mutation(self, DNA):
        for i in range(DNA_length):
            if random.random() < pm:
                DNA[0, i] = 1
            else:
                DNA[0, i] = 0
        return DNA

    #進化
    def evolve(self):
            DNA, DNA_id = self.selection() 
            DNA = self.crossover(DNA)     
            DNA = self.mutation(DNA)    
            self.populations[DNA_id] = DNA

ga=GA()
a=0
for step in range(generations):
    population_fitness = ga.calculate_fitness(ga.populations)
    x = ga.DNA_decode(ga.populations[np.argmax(population_fitness)])
    y_maximum = np.max(population_fitness)
    a=a+1
    print('X value : %.3f , maximum: %.3f , generations: %d' % (x, y_maximum,a)) 
    ga.evolve()
a_max =np.max(a)
#print('X value : %.3f , maximum: %.3f , generations: %d' % (x, y_maximum,a_max)) 