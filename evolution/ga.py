"""Genetic algorithm: tournament selection, elitism, uniform crossover, gaussian mutation."""
import numpy as np


def tournament(fit, rng, k=3):
    idx = rng.integers(0, len(fit), k)
    return idx[np.argmax(fit[idx])]


def next_generation(pop, fit, rng, elitism=4, mut_rate=0.1, mut_std=0.3, crossover=True, k=3):
    n, g = pop.shape
    order = np.argsort(-fit)
    new = [pop[i].copy() for i in order[:max(0, min(elitism, n))]]
    while len(new) < n:
        p1 = pop[tournament(fit, rng, k)]
        if crossover:
            p2 = pop[tournament(fit, rng, k)]
            mask = rng.random(g) < 0.5
            child = np.where(mask, p1, p2)
        else:
            child = p1.copy()
        m = rng.random(g) < mut_rate
        child = child + m * rng.normal(0, mut_std, g)
        new.append(child)
    return np.array(new)
