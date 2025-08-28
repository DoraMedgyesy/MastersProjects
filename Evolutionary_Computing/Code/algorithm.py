def EA(selection='Elitism', run_mode='train', enemy=8, run=1, best_run=0):

    print(f'\n{selection} vs Enemy {enemy}: Run {run}')

    from evoman.environment import Environment
    from demo_controller import player_controller
    import numpy as np
    import os
    # import time

    npop = 100
    gens = 20
    mutation = 0.1
    elite_size = 0.05  # Retain the top 5% individuals as elites
    n_hidden_neurons = 10
    dom_u, dom_l = 1, -1

    visuals = False
    experiment_name = f'{run_mode}/Enemy {enemy}/{selection}/{run_mode}_{run}'

    if not os.path.exists(experiment_name): os.makedirs(experiment_name)  # Create Folders
    if not visuals: os.environ["SDL_VIDEODRIVER"] = "dummy"  # Remove Visuals
    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                      enemies=[enemy],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=visuals,
                      randomini='yes')

    # default environment fitness is assumed for experiment
    env.state_to_log()  # checks environment state
    ####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
    # ini = time.time()  # sets time marker
    # genetic algorithm params
    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    def simulation(env, x):
        """runs simulation"""
        f, p, e, t = env.play(pcont=x)
        return f

    def norm(x, pfit_pop):
        """normalizes"""
        if (max(pfit_pop) - min(pfit_pop)) > 0:
            x_norm = (x - min(pfit_pop)) / (max(pfit_pop) - min(pfit_pop))
        else:
            x_norm = 0

        if x_norm <= 0:
            x_norm = 0.0000000001
        return x_norm

    def evaluate(x):
        """evaluation"""
        return np.array(list(map(lambda y: simulation(env, y), x)))

    def tournament(pop):
        """tournament"""
        c1 = np.random.randint(0, pop.shape[0], 1)
        c2 = np.random.randint(0, pop.shape[0], 1)

        if fit_pop[c1] > fit_pop[c2]:
            return pop[c1][0]
        else:
            return pop[c2][0]

    def limits(x):
        """limits"""
        if x > dom_u:
            return dom_u
        elif x < dom_l:
            return dom_l
        else:
            return x

    def crossover(pop):
        """crossover"""

        total_offspring = np.zeros((0, n_vars))

        for p in range(0, pop.shape[0], 2):
            p1 = tournament(pop)
            p2 = tournament(pop)

            n_offspring = np.random.randint(1, 3 + 1, 1)[0]
            offspring = np.zeros((n_offspring, n_vars))

            for f in range(0, n_offspring):

                cross_prop = np.random.uniform(0, 1)
                offspring[f] = p1 * cross_prop + p2 * (1 - cross_prop)

                # mutation
                for i in range(0, len(offspring[f])):
                    if np.random.uniform(0, 1) <= mutation:
                        offspring[f][i] = offspring[f][i] + np.random.normal(0, 1)

                offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

                total_offspring = np.vstack((total_offspring, offspring[f]))

        return total_offspring

    def doomsday(pop, fit_pop):
        """# kills the worst genomes, and replace with new best/random solutions"""
        worst = int(npop / 4)  # a quarter of the population
        order = np.argsort(fit_pop)
        orderasc = order[0:worst]

        for o in orderasc:
            for j in range(0, n_vars):
                pro = np.random.uniform(0, 1)
                if np.random.uniform(0, 1) <= pro:
                    pop[o][j] = np.random.uniform(dom_l, dom_u)  # random dna, uniform dist.
                else:
                    pop[o][j] = pop[order[-1:]][0][j]  # dna from best

            fit_pop[o] = evaluate([pop[o]])

        return pop, fit_pop

    def roulette_wheel_selection(pop, fit_pop):
        fit_pop_cp = fit_pop
        # avoiding negative probabilities, as fitness is ranges from negative numbers
        fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop_cp), fit_pop)))
        probs = fit_pop_norm / fit_pop_norm.sum()
        chosen = np.random.choice(pop.shape[0], npop, p=probs, replace=False)
        chosen = np.append(chosen[1:], best)
        return pop[chosen], fit_pop[chosen]

    def elitist_selection(pop, fit_pop, elite_size):
        num_elites = int(elite_size * npop)
        sorted_indices = np.argsort(fit_pop)[::-1]
        elites = pop[sorted_indices[:num_elites]]  # Select top individuals

        fit_pop_cp = fit_pop
        fit_pop_norm = np.array(list(map(lambda y: norm(y, fit_pop_cp), fit_pop)))  # Normalize fitness values
        probs = fit_pop_norm / np.sum(fit_pop_norm)  # Compute selection probabilities
        chosen = np.random.choice(pop.shape[0], npop - num_elites, p=probs, replace=False)  # Select the rest

        new_population = np.vstack((elites, pop[chosen]))
        new_fit_pop = np.append(fit_pop[sorted_indices[:num_elites]], fit_pop[chosen])

        return new_population, new_fit_pop

    def rank_selection(pop, fit_pop):
        sorted_indices = np.argsort(fit_pop)[::1]  # Sort fitness values in ascending order
        ranks = np.arange(1, len(fit_pop) + 1)  # Create Ranks from 1 to n
        probs = ranks / np.sum(ranks)  # Normalize ranks to get selection probabilities
        # Select individuals based on ranks
        chosen_indices = np.random.choice(sorted_indices, size=npop, replace=True, p=probs)
        return pop[chosen_indices], fit_pop[chosen_indices]

    # Test Best Solution
    if run_mode == 'test':
        best_solution = np.loadtxt(f'train/Enemy {enemy}/{selection}/train_{best_run}' + '/best.txt')
        env.update_parameter('speed', 'normal')
        evaluate([best_solution])
        gain = env.get_playerlife() - env.get_enemylife()
        print(f'Gain: {gain}')
        return gain

    # Start New Evolution
    if not os.path.exists(experiment_name + '/evoman_solstate'):
        pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
        fit_pop = evaluate(pop)
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        ini_g = 0
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)

    # Continue Evolution
    else:
        env.load_state()
        pop = env.solutions[0]
        fit_pop = env.solutions[1]
        best = np.argmax(fit_pop)
        mean = np.mean(fit_pop)
        std = np.std(fit_pop)
        # finds last generation number
        file_aux = open(experiment_name + '/gen.txt', 'r')
        ini_g = int(file_aux.readline())
        file_aux.close()

    # Save Results of pop 0
    file_aux = open(experiment_name + '/results.txt', 'a')
    file_aux.write('\n\ngen best mean std')
    file_aux.write(f'\n{ini_g} {fit_pop[best]:.6f} {mean:.6f} {std:.6f}')
    file_aux.close()

    # evolution
    last_sol = fit_pop[best]
    notimproved = 0

    # For Each Generation
    for i in range(ini_g + 1, gens):

        # Create Offspring
        offspring = crossover(pop)
        fit_offspring = evaluate(offspring)
        pop = np.vstack((pop, offspring))
        fit_pop = np.append(fit_pop, fit_offspring)

        # Find Best Individual
        best = np.argmax(fit_pop)
        fit_pop[best] = float(evaluate(np.array([pop[best]]))[0])  # repeats best eval, for stability issues
        best_sol = fit_pop[best]

        # Apply Selection
        if selection == 'Roulette_Wheel':
            pop, fit_pop = roulette_wheel_selection(pop, fit_pop)
        elif selection == 'Elitism':
            pop, fit_pop = elitist_selection(pop, fit_pop, elite_size=elite_size)
        elif selection == 'Ranked':
            pop, fit_pop = rank_selection(pop, fit_pop)

        # Check if Generations Are Improving
        if best_sol <= last_sol:
            notimproved += 1
        else:
            last_sol = best_sol
            notimproved = 0

        # NO? Activate Doomsday
        if notimproved >= 15:
            file_aux = open(experiment_name + '/results.txt', 'a')
            file_aux.write('\ndoomsday')
            file_aux.close()

            pop, fit_pop = doomsday(pop, fit_pop)
            notimproved = 0

        # Results
        best = np.argmax(fit_pop)
        std = np.std(fit_pop)
        mean = np.mean(fit_pop)

        # Save Results
        file_aux = open(experiment_name + '/results.txt', 'a')
        file_aux.write(f'\n{i} {fit_pop[best]:.6f} {mean:.6f} {std:.6f}')
        file_aux.close()

        # Save Generation Number
        file_aux = open(experiment_name + '/gen.txt', 'w')
        file_aux.write(str(i))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(experiment_name + '/best.txt', pop[best])

        # saves simulation state
        solutions = [pop, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

        print(f'Gen {i}')
    # fim = time.time()  # prints total execution time for experiment
    # print(f'\nExecution time: {round((fim - ini) / 60)} minutes \n')
    # print(f'\nExecution time: {round(fim - ini)} seconds \n')

    file = open(experiment_name + '/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
    file.close()

    env.state_to_log()  # checks environment state


if __name__ == "__main__":
    EA(run_mode='test')
