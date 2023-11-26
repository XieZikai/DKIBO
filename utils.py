import numpy as np
from deap import base
from deap import creator
from deap import tools
import random
import array


def round_result(x, constraint=None):
    if isinstance(x, dict):
        dict_keys = list(x.keys())
        dict_array = np.array(list(x.values()))
        dict_array_rounded = dict_array.round()

        def array_to_dict():
            rounded_dict = {}
            for i in range(len(dict_keys)):
                rounded_dict[dict_keys[i]] = dict_array_rounded[i]
            return rounded_dict

        if constraint is None:
            return array_to_dict()

        if constraint['type'] == 'eq':
            while constraint['fun'](dict_array_rounded) != 0:
                if constraint['fun'](dict_array_rounded) > 0:
                    i = np.argmax(dict_array_rounded - dict_array)
                    dict_array_rounded[i] -= 1
                if constraint['fun'](dict_array_rounded) < 0:
                    i = np.argmin(dict_array_rounded - dict_array)
                    dict_array_rounded[i] += 1
            return array_to_dict()

        if constraint['type'] == 'ineq':
            while constraint['fun'](dict_array_rounded) > 0:
                i = np.argmax(dict_array_rounded - dict_array)
                dict_array_rounded[i] -= 1
            return array_to_dict()
    else:
        dict_array = np.array(x)
        dict_array_rounded = dict_array.round()

        if constraint is None:
            return dict_array_rounded

        if constraint['type'] == 'eq':
            while constraint['fun'](dict_array_rounded) != 0:
                if constraint['fun'](dict_array_rounded) > 0:
                    i = np.argmax(dict_array_rounded - dict_array)
                    dict_array_rounded[i] -= 1
                if constraint['fun'](dict_array_rounded) < 0:
                    i = np.argmin(dict_array_rounded - dict_array)
                    dict_array_rounded[i] += 1
            return dict_array_rounded

        if constraint['type'] == 'ineq':
            while constraint['fun'](dict_array_rounded) > 0:
                i = np.argmax(dict_array_rounded - dict_array)
                dict_array_rounded[i] -= 1
            return dict_array_rounded

FirstCall = True


def uniform(bounds):
    return [random.uniform(b[0], b[1]) for b in bounds]


def all_elements_equal(lst, value):
    for element in lst:
        if element != value:
            return False
    return True


def NSGAII(objective, pbounds, target, constraint, seed=None, NGEN=100, MU=100, CXPB=0.9, max_delta=100000.0):
    random.seed(seed)
    weight_set = []
    for i in target:
        if i == 'max':
            weight_set.append(1.0)
        elif i == 'min':
            weight_set.append(-1.0)
        else:
            raise Exception('Optimization target needs to be either max or min')
    weight_set = tuple(weight_set)

    def feasible(x):
        if constraint(x) >= 0:
            return True
        return False

    global FirstCall
    if FirstCall:
        if all_elements_equal(target, 'max'):
            creator.create("FitnessMax", base.Fitness, weights=weight_set)
            creator.create("Individual", array.array, typecode='d',
                           fitness=creator.FitnessMax)
        elif all_elements_equal(target, 'min'):
            creator.create("FitnessMin", base.Fitness, weights=weight_set)
            creator.create("Individual", array.array, typecode='d',
                           fitness=creator.FitnessMin)
        else:
            creator.create("FitnessMulti", base.Fitness, weights=weight_set)
            creator.create("Individual", array.array, typecode='d',
                           fitness=creator.FitnessMulti)
        FirstCall = False
    toolbox = base.Toolbox()

    NDIM = len(pbounds)

    toolbox.register("attr_float", uniform, pbounds)

    toolbox.register("individual",
                     tools.initIterate,
                     creator.Individual,
                     toolbox.attr_float)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", objective)

    if constraint is not None:
        def distance(x):
            return [0.0 for _ in range(len(target))]
        delta = []
        for i in target:
            if i == 'max':
                delta.append(max_delta)
            else:
                delta.append(-max_delta)

        toolbox.decorate("evaluate", tools.DeltaPenality(feasible, delta, distance))

    toolbox.register("mate",
                     tools.cxSimulatedBinaryBounded,
                     low=pbounds[:, 0].tolist(),
                     up=pbounds[:, 1].tolist(),
                     eta=20.0)

    toolbox.register("mutate",
                     tools.mutPolynomialBounded,
                     low=pbounds[:, 0].tolist(),
                     up=pbounds[:, 1].tolist(),
                     eta=20.0,
                     indpb=1.0 / NDIM)

    toolbox.register("select", tools.selNSGA2)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=MU)

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)

    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)

            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        # print(logbook.stream)

    front = np.array([ind.fitness.values for ind in pop])

    return pop, logbook, front
