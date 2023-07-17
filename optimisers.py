class AIS:

    def __init__(self,
                 func,
                 size: int,
                 variables: int,
                 span: tuple,
                 clones: int,
                 mutation_decay: float,
                 ):

        self.SIZE = size
        self.VARS = variables
        self.FLOOR, self.CEIL = span
        self.MUTATION_SCALE = (self.CEIL - self.FLOOR) / 6
        self.CLONES = clones
        self.MUTATION_RATES = np.exp(-mutation_decay * (np.arange(self.SIZE, 0, -1)/self.SIZE))[:, np.newaxis, np.newaxis]

        self.FUNC = func
        self.GEN = 0
        
    
    def run(self, iters):
        
        for i in range(iters):
            score = self._step()
            self.GEN += 1
            np.save(f'ais/x_{self.GEN}', self.antibodies)
            np.save(f'ais/y_{self.GEN}', self.fitness)
        return score
        
    def load(self, gen:int):
        self.GEN = gen
        self.antibodies = np.load(f'ais/x_{self.GEN}.npy')
        self.fitness = np.load(f'ais/y_{self.GEN}.npy')

    def spawn(self):
        self.antibodies = (np.random.random((self.SIZE, self.VARS, 1)) * (self.CEIL - self.FLOOR) + self.FLOOR)
        self.fitness = np.expand_dims(np.apply_along_axis(self.FUNC, axis=1, arr=self.antibodies), axis=1)

        
    def _step(self):
        
        # select N best individuals
        
        source = np.squeeze(np.argsort(self.fitness, axis=0))
        
        # mutate clones INVERSELY PROPORTIONAL to their scores
        # alpha = exp(-p * f)
        #self.MUTATION_RATES = 0.1
        mutations = rng.normal(loc=0, scale=self.MUTATION_SCALE, size=(self.SIZE, self.VARS, self.CLONES))
        mutation_mask = rng.random((self.SIZE, self.VARS, self.CLONES)) < self.MUTATION_RATES
        
        clones = (mutations * mutation_mask) + self.antibodies[source]

        clones = np.clip(clones, self.FLOOR, self.CEIL)
        clones_scores = np.apply_along_axis(self.FUNC, axis=1, arr=clones)[:, None]
        clone_indices = np.nanargmin(clones_scores, axis=2)[..., None]
        
        clones = np.take_along_axis(clones, clone_indices, axis=2)
        clones_scores = np.take_along_axis(clones_scores, clone_indices, axis=2)
        
        all_antibodies = np.concatenate((self.antibodies, clones))
        all_scores = np.concatenate((self.fitness, clones_scores))

        best_indices = np.squeeze(np.argsort(all_scores, axis=0)[:self.SIZE])
        self.fitness = all_scores[best_indices]
        self.antibodies = all_antibodies[best_indices]
        
        return np.nanmin(self.fitness)






class GA:
    
    ###### INITIALISATION ######
    def __init__(self,
                 func,
                 size: int,
                 genes: int,
                 select: int,
                 tourn: int,
                 mutation: float,
                 span: tuple=(-1.0, 1.0),
                 ):

        #### POPULATION ATTRIBUTES ####
        self.SIZE = size
        self.GENES = genes
        self.FLOOR, self.CEIL = span
        assert select % 2 == 0
        self.SELECT = select
        self.TOURN = tourn

        self.MUTATION_RATE = mutation
        self.MUTATION_SCALE = (self.CEIL - self.FLOOR) / 6
        self.FUNC = func
        
        self.calls=0
        self.GEN = 0


    
    def run(self, iters):
        
        self.calls += self.SIZE
        for i in range(iters):
            score = self._step()
            self.GEN += 1
            np.save(f'ga/x_{self.GEN}', self.individuals)
            np.save(f'ga/y_{self.GEN}', self.fitness)
        return score
    
    def load(self, gen:int):
        self.GEN = gen
        self.individuals = np.load(f'ga/x_{self.GEN}.npy')
        self.fitness = np.load(f'ga/y_{self.GEN}.npy')
    
    def spawn(self):
        self.individuals = (np.random.random((self.SIZE, self.GENES)) * (self.CEIL - self.FLOOR) + self.FLOOR)
        self.fitness = np.expand_dims(np.apply_along_axis(self.FUNC, axis=1, arr=self.individuals), axis=1)


    def _step(self):
        
        # TOURNAMENT SELECTION (N = selection size, k = tourn size)
            # N * k matrix of random integers between 0, popsize - 1
            # matrix of scores by indexing score array
            # array of indices of best of each in k (argmin)
            # new selected matrix
        
        rand_indices = rng.integers(0, self.SIZE, (self.SELECT, self.TOURN))
        indices_scores = np.squeeze(self.fitness[rand_indices])
        selection = self.individuals[np.nanargmin(indices_scores, axis=1)]
                         
        # UNIFORM CROSSOVER
        rand_bools = rng.random((self.SELECT // 2, self.GENES)) > 0.5
        crossover_mask = np.concatenate((rand_bools, ~rand_bools))
        
        inverted = np.concatenate((selection[self.SELECT//2:,:], selection[:self.SELECT//2, :])) * (~crossover_mask)
        selection *= crossover_mask
        selection += inverted

        # MUTATION (r = mutation rate, s = mutation scale)
            # N * C matrix of uniform-distributed numbers between -s, s
            # N * C boolean mask with probability r
            # selection + (distribution * mask) 
        
        mutations = rng.normal(loc=0, scale=self.MUTATION_SCALE, size=(self.SELECT, self.GENES))
        mutation_mask = rng.random((self.SELECT, self.GENES)) < self.MUTATION_RATE
        selection += (mutations * mutation_mask)
        
        selection = np.clip(selection, self.FLOOR, self.CEIL)
        # RETAIN pop size BEST INDIVIDUALS
        self.calls += (self.SELECT * self.TOURN)
        selection_scores = np.expand_dims(np.apply_along_axis(self.FUNC, axis=1, arr=selection), axis=1)
        all_individuals = np.concatenate((self.individuals, selection))
        all_scores = np.concatenate((self.fitness, selection_scores))
        
        best_indices = np.squeeze(np.argsort(all_scores, axis=0)[:self.SIZE])
        self.fitness = all_scores[best_indices]
        self.individuals = all_individuals[best_indices]

        return np.nanmin(self.fitness)









    class PSO:

    ###### INITIALISATION ######
    def __init__(self,
                 size: int,
                 dims: int,
                 span: tuple=(-1.0, 1.0),
                 alpha: float = 0.5,    # velocity
                 beta: float = 0.5,     # personal best
                 gamma: float = 0.5,    # informant best
                 delta: float = 0.5,    # global best
                 epsilon: float = 1.0,  # jump size
                 sample: float = 0.1,   # informant proportion
                 #mode: str = 'clip'     # environment mode
                 ):
        assert min([size, alpha, beta, gamma, delta, epsilon]) >= 0 # ensure all weights are greater than or equal to 0


        #### SWARM ATTRIBUTES ####
        self.SIZE = size
        self.DIMS = dims
        self.FLOOR, self.CEIL = span


        #### SCALARS ####
        self.ALPHA = alpha  # velocity
        self.BETA = beta    # personal best
        self.GAMMA = gamma  # informant best
        self.DELTA = delta  # global best

        self.EPSILON = epsilon  # jump size

        self.SAMPLE = sample  # informant proportion

        self.GEN = 0

        self._spawn()   # generate swarm



    ###### PUBLIC RUN COMMAND ######
    # 'func' is a fitness function that operates on a 1D array to produce a single value.
    def run(self, func, iters):
        # return a list of each best value
        for i in range(iters):
            score = self._step(func)
            self.GEN += 1
            np.save(f'pso/i', self.informers)
            np.save(f'pso/x_{self.GEN}', self.particles)
            np.save(f'pso/v_{self.GEN}', self.velocity)
            np.save(f'pso/pb_{self.GEN}', self.personal_best)
            np.save(f'pso/pl_{self.GEN}', self.personal_location)

        return score

    def load(self, gen:int):
        self.GEN = gen
        self.informers = np.load(f'pso/i.npy')
        self.particles = np.load(f'pso/x_{self.GEN}.npy')
        self.velocity = np.load(f'pso/v_{self.GEN}.npy')
        self.personal_best = np.load(f'pso/pb_{self.GEN}.npy')
        self.personal_location = np.load(f'pso/pl_{self.GEN}.npy')


    ###### GENERATE PARTICLE SWARM ######
    def _spawn(self):

        #### PARTICLES ####
        # generate SIZE number of particles, each with the dimensionality of SHH, with coordinates between -span & span.
        self.particles = (rng.random((self.SIZE, self.DIMS)) * (self.CEIL - self.FLOOR) + self.FLOOR)

        #### VELOCITIES ####
        # generate similarly shaped velocities, with each being a SIZE-dimensional vector, between -1 & 1.
        self.velocity = (rng.random((self.SIZE, self.DIMS)) * 2) - 1

        #### INFORMER INDEXES ####
        # initialise a collection of random integers for each particle, to index the particle array as informants
        self.informers = rng.integers(0, self.SIZE, size=(self.SIZE, round(self.SIZE * self.SAMPLE)))
        #### RUN SETUP ####
        # track location
        self.personal_location = self.particles
        # set fitness to be unreasonably bad (huge number)
        # this will be reset by the first fitness test, but avoids the need for an extra check.
        # special values, like np.Infinity or np.NaN, will not produce 0 upon multiplication with 0, as is necessary.
        self.personal_best = np.full((self.SIZE, 1), 1e+100)



    ###### TIMESTEP ALGORITHM ######
    def _step(self, func):

        #### PARTICLE FITNESS ####
        # map fitness function over each particle
        new_personal = np.expand_dims(np.apply_along_axis(func, axis=1, arr=self.particles), axis=1)

        #### UPDATE PARTICLE BESTS ####
        # generate update mask
        updater = new_personal < self.personal_best  # outputs a boolean array
        # mask application replaces old, lesser values with new, greater ones in the array
        # apply mask to scores
        self.personal_best = (new_personal * updater) + (self.personal_best * (~updater))  # False values become 0
        # apply mask to location
        self.personal_location = (self.particles * updater) + (self.personal_location * (~ updater))


        #### CALCULATE INFORMANTS ####
        # get personal best scores for all informants
        informant_best = self.personal_best[self.informers]
        # find the coordinates of each group's best
        informant_location = np.squeeze(self.personal_location[np.nanargmin(informant_best, axis=1)], axis=1)

        #### CALCULATE GLOBAL ####
        # similarly find the coordinates of the global best
        global_location = self.personal_location[np.nanargmin(self.personal_best), :]

        #### CREATE UPDATE COMPONENTS ####
        # generate b, c & d scalar arrays by scaling random numbers to be between 0 & hyperparameters
        beta = rng.random((self.SIZE, self.DIMS)) * self.BETA
        gamma = rng.random((self.SIZE, self.DIMS)) * self.GAMMA
        delta = rng.random((self.SIZE, self.DIMS)) * self.DELTA

        # get directional vectors, and scale by b, c & d
        p = beta * (self.personal_location - self.particles)
        i = gamma * (informant_location - self.particles)
        g = delta * (global_location - self.particles)

        #### PERFORM PARTICLE UPDATE ####
        # generate final velocity array
        self.velocity = (self.ALPHA * self.velocity) + p + i + g
        # apply velocity array and environment rules to particles.
        self.particles = np.clip(self.particles + (self.EPSILON * self.velocity), self.FLOOR, self.CEIL)

        # return the best score this iteration.
        return np.nanmin(self.personal_best)
