import optimisers

from optproblems import cec2005


# Quick script to test metaheuristics.

# Hyperparameters are not particularly optimal.



prob = cec2005.F1(10) # global minimum is -450


# Artificial immune system
ais = optimisers.AIS(func=prob,
                     size=100,
                     variables=10,
                     span=(-100,100),
                     clones=25,
                     mutation_decay=0.9)

ais.spawn()

print(ais.run(100))



# Genetic algorithm
ga = optimisers.GA(func=prob,
                   size=100,
                   genes=10,
                   select=20,
                   tourn=10,
                   mutation=0.1,
                   span=(-100,100)
                   )

ga.spawn()

print(ga.run(100))


# Particle swarm optimisation
pso = optimisers.PSO(size=200,
                     dims=10,
                     span=(-100,100),
                     )

print(pso.run(prob, 100))

