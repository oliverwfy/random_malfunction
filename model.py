from utility import *
from sklearn.metrics import accuracy_score, precision_score, recall_score



def simulate_malfunctioning(simulation_times=100, pop_n=100, max_iteration=10000, k=3, init_x = 0.5, mal_x = None, equal_weights=False,
             alpha=0.5, prob_evidence=0.02, malfunctioning=0.0, threshold= 0.5, weights_updating = None, dampening=None,
             noise=None, pooling=True, memory=False,trend=True, memory_weight=0.5, consensus_only=False, detection_only=False):

    detection_time = []
    consensus_time = []
    accuracy = np.empty([max_iteration, simulation_times])
    precision = accuracy.copy()
    recall = accuracy.copy()

    accuracy_swarm = accuracy.copy()
    precision_swarm = accuracy.copy()
    recall_swarm = accuracy.copy()

    belief_avg_pred_good = accuracy.copy()
    belief_avg_true_good = accuracy.copy()

    print('--------------Simulation Starts--------------')
    print('Simulation times : {}'.format(simulation_times))
    print('Malfunctioning : {}'.format(malfunctioning))
    if mal_x:
        print(f'Malfunctioning Belief: {mal_x}\n')
    else:
        print(f'Malfunctioning Belief: Uni(0,1)\n')


    if not pooling:
        print('Evidence Only:')
        print('Params: alpha = {} \n'.format(alpha))

    else:
        print('Log-linear operator:')
        print('Params: k = {} alpha = {}\n'.format(k, alpha))

    for n in range(simulation_times):
        consensus = False
        detection = False
        pop = np.array([Agent(pop_n, _, init_x,  True) for _ in range(pop_n)])
        true_mal_id = generate_malfunctioning_agents(pop, malfunctioning, init_x=mal_x)

        true_mal_ls = np.zeros(pop_n)
        true_mal_ls[true_mal_id] = 1

        for i in range(max_iteration):
            # evidential updating
            for agent in pop:
                if agent.state:
                    if np.random.random() <= prob_evidence and agent.state:
                        if noise and np.random.random() <= noise:
                            evidential_update(agent, 1-alpha, dampening)
                        else:
                            evidential_update(agent, alpha, dampening)
                else:
                    agent.malfunction()

            if pooling:
                # individual good agent's detection
                accuracy[i,n] = accuracy_score(true_mal_ls, pop[-1].mal_detection())
                precision[i,n] = precision_score(true_mal_ls, pop[-1].mal_detection())
                recall[i,n] = recall_score(true_mal_ls, pop[-1].mal_detection())

                pred_mal_ls = np.zeros(pop_n)
                for agent in pop:
                    pred_mal_ls += agent.mal_detection()

                # an agent is malfunctioning if it is labeled by half of agents
                pred_mal_id = np.where(pred_mal_ls >= pop_n/2)[0]
                pred_mal_ls = np.zeros(pop_n)
                pred_mal_ls[pred_mal_id] = 1

                # swarm's prediction
                accuracy_swarm[i,n] = accuracy_score(true_mal_ls, pred_mal_ls)
                precision_swarm[i,n] = precision_score(true_mal_ls, pred_mal_ls)
                recall_swarm[i,n] = recall_score(true_mal_ls, pred_mal_ls)

                # average belief of "good" agents
                belief_avg_pred_good[i,n] = avg_belief_good(pred_mal_id, pop)
                belief_avg_true_good[i,n] = avg_belief_good(true_mal_id, pop)

                # opinion pooling
                pool = np.random.choice(pop, size=k, replace=False)

                if weights_updating == "beta":
                    opinion_pooling_beta(pool, threshold=threshold, memory=memory, lamb=memory_weight)
                elif weights_updating == "normal":
                    opinion_pooling_norm(pool, threshold=threshold, memory=memory, lamb=memory_weight)
                else:
                    if not equal_weights:
                        opinion_pooling(pool, threshold=threshold, time_step=i, trend=trend, memory=memory, lamb=memory_weight)
                    else:
                        opinion_pooling_equal_weights(pool, threshold=threshold, memory=memory, lamb=memory_weight)

                # check completion of fault detection
                if accuracy_swarm[i,n] == 1.0 and not detection:
                    detection_time.append(i)
                    detection = True

                    # terminates simulation if detection only
                    if detection_only:
                        break

            # check consensus
            if check_consensus(belief_avg_pred_good, i) and not consensus:
                consensus_time.append(i)
                consensus = True
                # terminate current simulation loop if consider consensus only
                if consensus_only:
                    break

        # add max iteration if not reach consensus
        if not consensus:
            consensus_time.append(max_iteration)
        # add max iteration if not detect all faults
        if not detection:
            detection_time.append(max_iteration)

    # average consensus time
    consensus_time_avg = np.array(consensus_time).mean()


    str_consensus_time_avg = f'Mean Consensus time = {consensus_time_avg}'

    if int(consensus_time_avg) == max_iteration:
        str_consensus_time_avg += ' (no consensus)'

    print(str_consensus_time_avg)
    print('----------------Simulation ends----------------\n\n')

    result = {'consensus_time': np.array(consensus_time), 'detection_time' : np.array(detection_time),
              'accuracy': accuracy, 'precision': precision, 'recall': recall,
              'accuracy_avg': accuracy_swarm, 'precision_avg': precision_swarm, 'recall_avg': recall_swarm,
              'belief_avg_pred_good': belief_avg_pred_good, 'belief_avg_true_good': belief_avg_true_good,
              'belief_dict': pop[-1].belief_dict}

    return result




def simulate_malicious(simulation_times=100, pop_n=100, max_iteration=10000, k=3, init_x = 0.5, classification = True, dampening=None,
             alpha=0.5, prob_evidence=0.02, malicious=0.01, mal_c=None,strategy='deception', threshold= 0.5, weights_updating = None,
             noise=None, pooling=True, memory=False, memory_weight=0.5, consensus_only=False, detection_only=False):

    # detection_time = []
    # consensus_time = []
    # accuracy = np.empty([max_iteration, simulation_times])
    # precision = accuracy.copy()
    # recall = accuracy.copy()

    # accuracy_swarm = accuracy.copy()
    # precision_swarm = accuracy.copy()
    # recall_swarm = accuracy.copy()

    # belief_avg_pred_good = accuracy.copy()
    belief_avg_true_good = np.empty([max_iteration, simulation_times])

    print('--------------Simulation Starts--------------')
    print('Simulation times : {}'.format(simulation_times))

    print('Malicious : {}  c : {}'.format(malicious, mal_c))


    if not pooling:
        print('Evidence Only:')
        print('Params: alpha = {} \n'.format(alpha))

    else:
        print('Log-linear operator:')
        print('Params: k = {} alpha = {}\n'.format(k, alpha))

    for n in range(simulation_times):
        consensus = False
        detection = False
        pop = np.array([Agent(pop_n, _, init_x,  True) for _ in range(pop_n)])
        true_mal_id = generate_malicious_agents(pop, malicious, threshold, mal_c)
        true_mal_ls = np.zeros(pop_n)
        true_mal_ls[true_mal_id] = 1

        for i in range(max_iteration):

            # evidential updating
            for agent in pop:
                if np.random.random() <= prob_evidence and agent.state:
                    if noise and np.random.random() <= noise:
                        evidential_update(agent, 1-alpha, dampening)
                    else:
                        evidential_update(agent, alpha, dampening)

            if pooling:
                # if classification:
                #     # individual good agent's detection
                #     accuracy[i,n] = accuracy_score(true_mal_ls, pop[-1].mal_detection())
                #     precision[i,n] = precision_score(true_mal_ls, pop[-1].mal_detection())
                #     recall[i,n] = recall_score(true_mal_ls, pop[-1].mal_detection())
                #     pred_mal_ls = np.zeros(pop_n)
                #
                #     for agent in pop:
                #         pred_mal_ls += agent.mal_detection()
                #
                #     # an agent is malfunctioning if it is labeled by half of agents
                #     pred_mal_id = np.where(pred_mal_ls >= pop_n/2)[0]
                #     pred_mal_ls = np.zeros(pop_n)
                #     pred_mal_ls[pred_mal_id] = 1
                #
                #     # swarm's prediction
                #     accuracy_swarm[i,n] = accuracy_score(true_mal_ls, pred_mal_ls)
                #     precision_swarm[i,n] = precision_score(true_mal_ls, pred_mal_ls)
                #     recall_swarm[i,n] = recall_score(true_mal_ls, pred_mal_ls)
                #
                #     # average belief of "good" agents
                #     belief_avg_pred_good[i,n] = avg_belief_good(pred_mal_id, pop)

                belief_avg_true_good[i,n] = avg_belief_good(true_mal_id, pop)

                # opinion pooling
                pool = np.random.choice(pop, size=k, replace=False)

                if weights_updating == "beta":
                    opinion_pooling_beta(pool, threshold=threshold, memory=memory, lamb=memory_weight)
                elif weights_updating == "normal":
                    opinion_pooling_norm(pool, threshold=threshold, memory=memory, lamb=memory_weight)
                else:
                    opinion_pooling_malicious(pool, threshold=threshold, strategy=strategy,memory=memory, lamb=memory_weight)

    #             # check completion of fault detection
    #             if accuracy_swarm[i,n] == 1.0 and not detection:
    #                 detection_time.append(i)
    #                 detection = True
    #                 # terminates simulation if detection only
    #                 if detection_only:
    #                     break
    #
    #         # check consensus
    #         if check_consensus(belief_avg_pred_good, i) and not consensus:
    #             consensus_time.append(i)
    #             consensus = True
    #             # terminate current simulation loop if consider consensus only
    #             if consensus_only:
    #                 break
    #
    #     # add max iteration if not reach consensus
    #     if not consensus:
    #         consensus_time.append(max_iteration)
    #     # add max iteration if not detect all faults
    #     if not detection:
    #         detection_time.append(max_iteration)
    #
    # # average consensus time
    # consensus_time_avg = np.array(consensus_time).mean()

    # str_consensus_time_avg = f'Mean Consensus time = {consensus_time_avg}'
    #
    # if int(consensus_time_avg) == max_iteration:
    #     str_consensus_time_avg += ' (no consensus)'

    # print(str_consensus_time_avg)

    print('----------------Simulation ends----------------\n\n')

    result = {'belief_avg_true_good': belief_avg_true_good, 'pop' : pop}

    # result = {'consensus_time': np.array(consensus_time), 'detection_time' : np.array(detection_time),
    #           'accuracy': accuracy, 'precision': precision, 'recall': recall,
    #           'accuracy_avg': accuracy_swarm, 'precision_avg': precision_swarm, 'recall_avg': recall_swarm,
    #           'belief_avg_pred_good': belief_avg_pred_good, 'belief_avg_true_good': belief_avg_true_good,
    #           'pop': pop}



    return result




def simulate_malicious_acc(simulation_times=100, pop_n=100, max_iteration=10000, k=3, init_x = 0.5, classification = True, dampening=None,
                       alpha=0.5, prob_evidence=0.02, malicious=0.01, mal_c=None,strategy='deception', threshold= 0.5, weights_updating = None,
                       noise=None, pooling=True, memory=False, memory_weight=0.5, consensus_only=False, detection_only=False):

    # detection_time = []
    # consensus_time = []
    accuracy = np.empty([max_iteration, simulation_times])
    # precision = accuracy.copy()
    # recall = accuracy.copy()

    accuracy_swarm = accuracy.copy()
    # precision_swarm = accuracy.copy()
    # recall_swarm = accuracy.copy()

    # belief_avg_pred_good = accuracy.copy()
    belief_avg_true_good = np.empty([max_iteration, simulation_times])

    print('--------------Simulation Starts--------------')
    print('Simulation times : {}'.format(simulation_times))

    print('Malicious : {}  c : {}'.format(malicious, mal_c))


    if not pooling:
        print('Evidence Only:')
        print('Params: alpha = {} \n'.format(alpha))

    else:
        print('Log-linear operator:')
        print('Params: k = {} alpha = {}\n'.format(k, alpha))

    for n in range(simulation_times):
        consensus = False
        detection = False
        pop = np.array([Agent(pop_n, _, init_x,  True) for _ in range(pop_n)])
        true_mal_id = generate_malicious_agents(pop, malicious, threshold, mal_c)
        true_mal_ls = np.zeros(pop_n)
        true_mal_ls[true_mal_id] = 1

        for i in range(max_iteration):

            # evidential updating
            for agent in pop:
                if np.random.random() <= prob_evidence and agent.state:
                    if noise and np.random.random() <= noise:
                        evidential_update(agent, 1-alpha, dampening)
                    else:
                        evidential_update(agent, alpha, dampening)

            if pooling:
                if classification:
                    # individual good agent's detection
                    # accuracy[i,n] = accuracy_score(true_mal_ls, pop[-1].mal_detection())
                    # precision[i,n] = precision_score(true_mal_ls, pop[-1].mal_detection())
                    # recall[i,n] = recall_score(true_mal_ls, pop[-1].mal_detection())
                    pred_mal_ls = np.zeros(pop_n)

                    for agent in pop:
                        pred_mal_ls += agent.mal_detection()

                    # an agent is malfunctioning if it is labeled by half of agents
                    pred_mal_id = np.where(pred_mal_ls >= pop_n/2)[0]
                    pred_mal_ls = np.zeros(pop_n)
                    pred_mal_ls[pred_mal_id] = 1

                    # swarm's prediction
                    accuracy_swarm[i,n] = accuracy_score(true_mal_ls, pred_mal_ls)
                    # precision_swarm[i,n] = precision_score(true_mal_ls, pred_mal_ls)
                    # recall_swarm[i,n] = recall_score(true_mal_ls, pred_mal_ls)

                    # average belief of "good" agents
                    # belief_avg_pred_good[i,n] = avg_belief_good(pred_mal_id, pop)

                belief_avg_true_good[i,n] = avg_belief_good(true_mal_id, pop)

                # opinion pooling
                pool = np.random.choice(pop, size=k, replace=False)

                if weights_updating == "beta":
                    opinion_pooling_beta(pool, threshold=threshold, memory=memory, lamb=memory_weight)
                elif weights_updating == "normal":
                    opinion_pooling_norm(pool, threshold=threshold, memory=memory, lamb=memory_weight)
                else:
                    opinion_pooling_malicious(pool, threshold=threshold, strategy=strategy,memory=memory, lamb=memory_weight)

    #             # check completion of fault detection
    #             if accuracy_swarm[i,n] == 1.0 and not detection:
    #                 detection_time.append(i)
    #                 detection = True
    #                 # terminates simulation if detection only
    #                 if detection_only:
    #                     break
    #
    #         # check consensus
    #         if check_consensus(belief_avg_pred_good, i) and not consensus:
    #             consensus_time.append(i)
    #             consensus = True
    #             # terminate current simulation loop if consider consensus only
    #             if consensus_only:
    #                 break
    #
    #     # add max iteration if not reach consensus
    #     if not consensus:
    #         consensus_time.append(max_iteration)
    #     # add max iteration if not detect all faults
    #     if not detection:
    #         detection_time.append(max_iteration)
    #
    # # average consensus time
    # consensus_time_avg = np.array(consensus_time).mean()

    # str_consensus_time_avg = f'Mean Consensus time = {consensus_time_avg}'
    #
    # if int(consensus_time_avg) == max_iteration:
    #     str_consensus_time_avg += ' (no consensus)'

    # print(str_consensus_time_avg)

    print('----------------Simulation ends----------------\n\n')


    # result = {'consensus_time': np.array(consensus_time), 'detection_time' : np.array(detection_time),
    #           'accuracy': accuracy, 'precision': precision, 'recall': recall,
    #           'accuracy_avg': accuracy_swarm, 'precision_avg': precision_swarm, 'recall_avg': recall_swarm,
    #           'belief_avg_pred_good': belief_avg_pred_good, 'belief_avg_true_good': belief_avg_true_good,
    #           'pop': pop}

    result= {'accuracy_avg': accuracy_swarm, 'belief_avg_true_good': belief_avg_true_good,
             'pop': pop}


    return result




