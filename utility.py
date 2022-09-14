import pandas as pd
from scipy.stats import beta, norm
from agent import *
from sklearn.linear_model import LinearRegression


tol = 10**-6


def confidence_interval(data_ls):
    df = pd.DataFrame(data_ls)
    l_quantile = df.quantile(0.025, axis=1)
    r_quantile = df.quantile(0.975, axis=1)

    mean = df.mean(axis=1)
    return [mean-l_quantile, r_quantile-mean]


def evidential_update(agent, alpha=0.5, dampening=None):
    factor = (1-alpha)*agent.x
    agent.x = factor / (factor + alpha*(1-agent.x))
    if dampening:
        agent.x = dampening/2 + (1-dampening)*agent.x

    return agent.x


def check_consensus(avg_confidence , ite=-1):

    return avg_confidence[ite][-1] >= 0.99


def my_kl_divergence(x1, x2):

    if x1 == 1.0:
        if x2 == 1.0:
            kl_div = 0
        else:
            if x2 == 0.0:
                kl_div = 1/tol
            else:
                kl_div = x1*np.log(x1/x2)
    else:
        if x2 == 1.0:
            if x1 == 0.0:
                kl_div = 1/tol
            else:
                kl_div = x1*np.log((x1)/x2)

        elif x2 == 0.0:
            kl_div = (1-x1)*np.log((1-x1)/(1-x2))
        else:
            kl_div = x1*np.log((x1)/x2) + (1-x1)*np.log((1-x1)/(1-x2))

    return kl_div

def kl_divergence(x1, x2):
    return entropy([x1, 1-x1], [x2, 1-x2])


# def total_variation_distance(x1, x2):
#     tv_distance = np.abs(x1-x2)
#     return tv_distance


# def softmax(input):
#     output = np.exp(input) / np.exp(input).sum()
#     return output


def weights_rescale(agent, pool_id, euqal_weights = False):

    confidence = agent.confidence[pool_id]

    if not euqal_weights:
        if confidence.sum() < tol:
            return np.zeros((len(confidence)))
        weight = confidence / confidence.sum()
    else:
        if confidence.sum() < tol:
            return 0.0
        return 1.0/confidence.sum()

    return np.array(weight)


def confidence_updating_beta(pool_prob):

    mean = pool_prob.mean()
    var = pool_prob.var()

    a = mean*(mean*(1-mean)/var - 1)
    b = a*(1-mean)/mean

    weight_arr = beta.pdf(pool_prob,a, b)
    return weight_arr / weight_arr.max()


def confidence_updating_norm(pool_prob):
    if np.all(pool_prob == pool_prob[0]):
        return np.ones_like(pool_prob)

    mean = pool_prob.mean()
    std = pool_prob.std()
    weight_arr = norm(mean, std).pdf(pool_prob)

    return weight_arr / weight_arr.max()


def confidence_updating(pool_prob, pooled_prob):
    # return np.exp(-(np.array([my_kl_divergence(pooled_prob, prob) for prob in pool_prob])))
    return np.exp(-(np.array([kl_divergence(pooled_prob, prob) for prob in pool_prob])))


def confidence_updating_with_trend(pool_prob, pooled_prob, pool_trend, pooled_trend):

    d_belief = np.array([kl_divergence(pooled_prob, prob) for prob in pool_prob])


    # d_trend = np.empty(len(d_belief))
    # for i, l in enumerate(pool_trend):
    #     l *= pooled_trend
    #     l[l < 0] *= 1.2
    #     d_trend[i] = np.sum(l) + 5
    #
    # d_trend = 2 * sigmoid(-d_trend)

    # d_trend = 2 * sigmoid(-np.sign(pooled_trend) * (pool_trend - pooled_trend))
    # d_trend = trend_mapping(-np.sign(pooled_trend) * (pool_trend - pooled_trend), 2)

    avg_belief = np.mean(pool_trend)
    d_trend = 2*sigmoid(-np.sign(pooled_trend) * (pool_trend - avg_belief))
    d = d_belief * d_trend

    return np.exp(-d)


def confidence_updating_with_trend_lr(pool_prob, pooled_prob, pool_trend, pooled_trend):

    d_belief = np.array([kl_divergence(pooled_prob, prob) for prob in pool_prob])


    d_trend = pool_trend - np.abs(pooled_trend)
    d_trend = 2 * sigmoid(-d_trend)
    d = d_belief * d_trend

    return np.exp(-d)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def trend_mapping(x, l):
    x[np.logical_and(x>=-l,x<=l)] = 1
    x[x>l] += 1-l
    x[x<-l] += 1+l
    return x


def log_op(x, w):
    numerator = np.prod(x**w)
    return numerator / (numerator + np.prod((1 - x) ** w))


def s_prod(x, w):
    numerator = np.prod(x) ** w
    return numerator / (numerator + np.prod(1-x) ** w)


def find_trend(belief_dict, pool_id, weights):
    pool_belief_dict = [belief_dict[id] for id in pool_id]
    avg_belief = []
    for b_dict in pool_belief_dict:
        avg_belief.append( np.mean([v for _,v in sorted(b_dict.items())]) )
    pooled_trend = np.dot([np.subtract(d[max(d.keys())], 0.5) for d in pool_belief_dict], weights)
    return np.array(avg_belief), np.sign(pooled_trend)


def trend_estimation(belief_dict, pool_id, weights):
    lr = LinearRegression(fit_intercept=False)
    pool_belief_dict = [np.array(belief_dict[id]) for id in pool_id]
    pool_trend = np.array([lr.fit(np.arange(len(d)).reshape(-1,1), (d-0.5)).coef_[0] for d in pool_belief_dict])
    pooled_trend = np.sum(pool_trend)/len(pool_trend)
    return np.sign(pooled_trend)*pool_trend, pooled_trend


def opinion_pooling(pool, threshold, time_step, trend, memory, lamb):
    pool_prob = np.array([agent.x for agent in pool])
    pool_id = np.array([agent.id for agent in pool])
    if not trend:
        for individual in pool:
            if individual.state:
                individual.update_belief_dict(pool_id, pool_prob)

                # rescale confidence (sum to 1)
                weights = weights_rescale(individual, pool_id)

                # log-linear operator
                pooled_prob = log_op(pool_prob, weights)

                # new confidence
                confidence_new = confidence_updating(pool_prob, pooled_prob)

                if memory:
                    # old confidence
                    confidence_old = individual.confidence[pool_id]
                    # weighted new confidence
                    confidence_new = (1-lamb)*confidence_old + lamb*confidence_new

                # if any predicted malfunctioning agents
                if (confidence_new < threshold).any():

                    confidence_new[confidence_new<threshold] = 0
                    individual.confidence[pool_id] = confidence_new
                    pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))
                    individual.x = pooled_prob

                else:

                    individual.x = pooled_prob
                    individual.confidence[pool_id] = confidence_new
            individual.mal_detection()

    else:
        pool_belief_dict = {id: {} for id in pool_id}
        for individual in pool:
            if individual.state:
                pool_belief_dict.update({key: individual.belief_dict.get(key) for key in pool_id})

        for individual in pool:
            if individual.state:
                individual.update_belief_dict(pool_id, time_step, pool_prob, pool_belief_dict)

                # rescale confidence (sum to 1)
                weights = weights_rescale(individual, pool_id)

                # log-linear operator
                pooled_prob = log_op(pool_prob, weights)

                # agents' trend and pooled trend
                # pool_trend, pooled_trend = find_trend(individual.belief_dict, pool_id, weights)
                avg_belief, pooled_trend = find_trend(individual.belief_dict, pool_id, weights)
                # pool_trend, pooled_trend = trend_estimation(individual.belief_dict, pool_id, weights)

                # new confidence
                # confidence_new = confidence_updating_with_trend_lr(pool_prob, pooled_prob, pool_trend, pooled_trend)
                confidence_new = confidence_updating_with_trend(pool_prob, pooled_prob, avg_belief, pooled_trend)

                if memory:
                    # old confidence
                    confidence_old = individual.confidence[pool_id]
                    # weighted new confidence
                    confidence_new = (1-lamb)*confidence_old + lamb*confidence_new

                # if any predicted malfunctioning agents
                if (confidence_new < threshold).any():

                    confidence_new[confidence_new<threshold] = 0
                    individual.confidence[pool_id] = confidence_new
                    pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))
                    individual.x = pooled_prob

                else:

                    individual.x = pooled_prob
                    individual.confidence[pool_id] = confidence_new

        individual.mal_detection()

    return None


def opinion_pooling_equal_weights(pool, threshold, memory, lamb):
    pool_prob = np.array([agent.x for agent in pool])
    pool_id = np.array([agent.id for agent in pool])
    for individual in pool:
        if individual.state:

            # rescale confidence (sum to 1)
            weights = weights_rescale(individual, pool_id, euqal_weights=True)

            if weights == 0.0:
                pooled_prob = individual.x
            else:
                # log-linear operator with same weights
                pooled_prob = s_prod(pool_prob, weights)

            # new confidence
            confidence_new = confidence_updating(pool_prob, pooled_prob)

            if memory:
                # old confidence
                confidence_old = individual.confidence[pool_id]
                # weighted new confidence
                confidence_new = (1-lamb)*confidence_old + lamb*confidence_new

            # if any predicted malfunctioning agents
            if (confidence_new < threshold).any():

                mal_id = confidence_new < threshold
                confidence_new[mal_id] = 0.0
                confidence_new[~mal_id] = 1.0
                individual.confidence[pool_id] = confidence_new
                pooled_prob = s_prod(pool_prob, weights_rescale(individual, pool_id, euqal_weights=True))
                individual.x = pooled_prob

            else:

                individual.x = pooled_prob
                individual.confidence[pool_id] = 1.0

        individual.mal_detection()

    return None




def opinion_pooling_malicious(pool, threshold,strategy, memory, lamb):


    pool_id = np.array([agent.id for agent in pool])
    loc_malicious = [i for i, agent in enumerate(pool) if not agent.state]

    if not loc_malicious:
        pool_prob = np.array([agent.x for agent in pool])
    else:
        pool_prob = np.empty(len(pool))
        for i, agent in enumerate(pool):
            if i in loc_malicious:
                pool_prob[i] = 0
            else:
                pool_prob[i] = agent.x

    for individual in pool:
        if individual.state:

            # rescale confidence (sum to 1)
            weights = weights_rescale(individual, pool_id)

            if loc_malicious:
                if strategy == 'deception':
                    # deception
                    x_m, pooled_prob = pool[loc_malicious[0]].deception(np.delete(pool_prob,loc_malicious),np.delete(weights,loc_malicious))
                else:
                    # pull only
                    x_m, pooled_prob = pool[loc_malicious[0]].pull_only(np.delete(pool_prob,loc_malicious),np.delete(weights,loc_malicious))

                for agent in pool[loc_malicious]:
                    agent.x = x_m

                pool_prob[loc_malicious] = x_m
            else:
                # no malicious agents
                pooled_prob = log_op(pool_prob, weights)

            # log linear
            pooled_prob = log_op(pool_prob, weights)
            # new confidence
            confidence_new = confidence_updating(pool_prob, pooled_prob)

            if memory:
                # old confidence
                confidence_old = individual.confidence[pool_id]
                # weighted new confidence
                confidence_new = (1-lamb)*confidence_old + lamb*confidence_new

            # if any predicted malfunctioning agents
            if (confidence_new < threshold).any():
                mal_id = pool_id[np.where(confidence_new<threshold)[0]]
                confidence_new[confidence_new<threshold] = 0
                individual.confidence[pool_id] = confidence_new
                pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))

                confidence_new = confidence_updating(pool_prob, pooled_prob)

                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new
                individual.confidence[mal_id] = 0

            else:
                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new

        individual.mal_detection()

    return


def opinion_pooling_beta(pool, threshold, memory, lamb):
    pool_prob = np.array([agent.x for agent in pool])
    pool_id = np.array([agent.id for agent in pool])
    for individual in pool:
        if individual.state:

            # rescale confidence (sum to 1)
            weights = weights_rescale(individual, pool_id)

            # log-linear operator
            pooled_prob = log_op(pool_prob, weights)

            # new confidence
            confidence_new = confidence_updating_beta(pool_prob)
            if memory:
                # old confidence
                confidence_old = individual.confidence[pool_id]
                # weighted new confidence
                confidence_new = (1-lamb)*confidence_old + lamb*confidence_new

            # if any predicted malfunctioning agents
            if (confidence_new < threshold).any():
                mal_id = pool_id[np.where(confidence_new<threshold)[0]]
                confidence_new[confidence_new<threshold] = 0
                individual.confidence[pool_id] = confidence_new
                pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))

                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new
                individual.confidence[mal_id] = 0

            else:
                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new

        individual.mal_detection()

    return None


def opinion_pooling_norm(pool, threshold, memory, lamb):
    pool_prob = np.array([agent.x for agent in pool])
    pool_id = np.array([agent.id for agent in pool])
    for individual in pool:
        if individual.state:

            # rescale confidence (sum to 1)
            weights = weights_rescale(individual, pool_id)

            # log-linear operator
            pooled_prob = log_op(pool_prob, weights)

            # new confidence
            confidence_new = confidence_updating_norm(pool_prob)

            if memory:
                # old confidence
                confidence_old = individual.confidence[pool_id]
                # weighted new confidence
                confidence_new = (1-lamb)*confidence_old + lamb*confidence_new

            # if any predicted malfunctioning agents
            if (confidence_new < threshold).any():
                mal_id = pool_id[np.where(confidence_new<threshold)[0]]
                confidence_new[confidence_new<threshold] = 0
                individual.confidence[pool_id] = confidence_new
                pooled_prob = log_op(pool_prob, weights_rescale(individual, pool_id))

                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new
                individual.confidence[mal_id] = 0

            else:
                individual.x = pooled_prob
                individual.confidence[pool_id] = confidence_new

        individual.mal_detection()

    return None

def generate_malicious_agents(pop, malicious, threshold, mal_c):
    malicious_id = []
    if malicious:
        malicious_id = np.arange(0, int(len(pop)*malicious))
        for i in malicious_id:
            pop[i] = Malicious(len(pop), pop[i].id, threshold, mal_c)

    return malicious_id

def generate_malfunctioning_agents(pop, malfunctioning, init_x=None):

    malfunctioning_id = []
    if malfunctioning:

        # malfunctioning_id = np.arange(0, len(pop), int(1/malfunctioning))
        malfunctioning_id = np.arange(0, int(len(pop)*malfunctioning))
        malfunctioning_agents = pop[malfunctioning_id]

        for mal_agent in malfunctioning_agents:
            mal_agent.state = False
            if not init_x:
                mal_agent.x = np.random.random()
            else:
                mal_agent.x = init_x

    return malfunctioning_id


def avg_belief_good(mal_id, pop):

    good_ids = list(set(range(len(pop))) - set(mal_id))
    good_agents = pop[good_ids]
    belief_avg = np.sum([agent.x for agent in good_agents]) / len(good_agents)

    return belief_avg
