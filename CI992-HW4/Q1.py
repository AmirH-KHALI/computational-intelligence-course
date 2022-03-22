import random 

def percentage(percent, whole):
  return (percent * whole) / 100.0

POPULATION_SIZE = 20
GENERATION_SIZE = 50
OO = 99999999999

city_count = 7
edges = [
         [ OO, 12, 10, OO, OO, OO, 12 ],
         [ 12, OO,  8, 12, OO, OO, OO ], 
         [ 10,  8, OO, 11,  3, OO,  9 ],
         [ OO, 12, 11, OO, 11, 10, OO ],
         [ OO, OO,  3, 11, OO,  6,  7 ],
         [ OO, OO, OO, 10,  7, OO,  9 ],
         [ 12, OO,  9, OO,  7,  9, OO ]
         ]

class GA: 
    
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_sum()

    def calculate_sum(self): 
        sum = 0
        for i in range(len(self.chromosome) - 1):
            sum += edges[self.chromosome[i]][self.chromosome[i + 1]]
        return sum

    def mutated_genes(self,child): 
        geneT1 = random.randint(1, len(self.chromosome) - 2)
        geneT2 = random.randint(1, len(self.chromosome) - 2)

        child[geneT1], child[geneT2] = child[geneT2], child[geneT1]
        
        return child
  
    def mate(self, other):
        childP1 = []
        childP2 = []

        startGene = random.randint(0, len(self.chromosome) - 2)
        endGene = random.randint(startGene + 1, len(self.chromosome) - 1)

        childP1 = self.chromosome[startGene:endGene]

        for c in other.chromosome:
            if c not in childP1 and c not in childP2:
                if len(childP2) < startGene:
                    childP2.append(c)
                else:
                    childP1.append(c)

        child = childP2 + childP1
        child.append(0)

        if random.random() < 0.01:
            child = self.mutated_genes(child)

        return GA(child)

population = []
for _ in range(POPULATION_SIZE):
    random_gnome = [0]
    for _ in range(city_count - 1):
        while (True):
            r = random.randint(1, city_count - 1)
            if r not in random_gnome:
                random_gnome.append(r)
                break;

    random_gnome.append(0)
    population.append(GA(random_gnome)) 

ans = None
for _ in range(GENERATION_SIZE):
    population = sorted(population, key = lambda x:x.fitness) 
    
    # print('Gen', _)
    # for p in population:
    #     print(p.chromosome, ':', p.fitness)

    ans = population[0].fitness

    count = 0 
    for i in range(len(population)):
        if population[i].fitness == ans:
            count += 1

    if count >= percentage(90, POPULATION_SIZE):
        break

    next_generation = [] 
 
    next_generation.extend(
        population[:int(percentage(30, POPULATION_SIZE))]
        ) 

    for _ in range(int(percentage(70, POPULATION_SIZE))): 
        
        half = int(percentage(50, POPULATION_SIZE))
        
        p1 = random.choice(population[:half])
        p2 = random.choice(population[:half])

        child = p1.mate(p2)
        next_generation.append(child)

    population = next_generation


print("Path :", population[0].chromosome, ", Cost :", population[0].fitness)


