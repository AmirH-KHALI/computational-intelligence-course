import random

POPULATION_SIZE = 200
GENERATION_SIZE = 100

def percentage(percent, whole):
  return (percent * whole) / 100.0

class GA: 

    def __init__(self, chromosome): 
        self.chromosome = chromosome  
        self.fitness = self.calculate_fitness() 
  
    def calculate_fitness(self):
        s = str(self.chromosome)
        
        decimal = int(s[2:9], 2)
        floating = int(s[9:], 2)
        x = decimal + (floating / (10 ** len(str(floating))))
        
        if s[1] == "1":
            x = -x
        
        self.x = x
        return abs((9 * (x ** 5)) - (194.7 * (x ** 4)) + (1680.1 * (x ** 3)) - (7227.94 * (x ** 2)) + (15501.2 * x) - 13257.2)

    def mutated_genes(self,child): 
        geneT = random.randint(1, len(child) - 1)

        if child[geneT] == "0":
            child[geneT] = "1"
        else:
            child[geneT] = "0"

        return child
  
    def mate(self, other): 
        p1 = list(str(self.chromosome))
        p2 = list(str(other.chromosome))
        
        startGene = random.randint(0, len(p1) - 2)
        endGene = random.randint(startGene + 1, len(p2) - 1)

        p2[startGene:endGene] = p1[startGene:endGene]
        child = p2
        
        if random.random() < 0.05:
            child = self.mutated_genes(child)
        
        return GA(int("".join(child)))

population = []
for _ in range(POPULATION_SIZE):
    random_gnome = '1'
    for _ in range(15):
        random_gnome += str(random.randint(0,1))
    random_gnome = int(random_gnome)
    population.append(GA(random_gnome)) 

for _ in range(GENERATION_SIZE):
    population = sorted(population, key = lambda x:x.fitness) 

    # print('Gen', _)
    # for p in population:
    #     print(p.chromosome, ':', p.fitness)

    if population[0].fitness < 0.02:
        break

    next_generation = []
    next_generation.extend(population[:int(percentage(30, POPULATION_SIZE))]) 

    for _ in range(int(percentage(70, POPULATION_SIZE))):
        
        half = int(percentage(50, POPULATION_SIZE))

        p1 = random.choice(population[:half])
        p2 = random.choice(population[:half])

        child = p1.mate(p2) 
        next_generation.append(child) 

    population = next_generation 


print('X :', population[0].x, ', Score :', population[0].fitness)


