#Consider the following program that uses a hill-climbing algorithm to 
# find hyperparameters(alpha, eta) for a perceptron model
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from numpy.random import rand, randn
from numpy import mean


def obj_func(X, y, cfg):
    # unpacking the hyperparameters
    eta, alpha = cfg
    model = Perceptron(penalty='elasticnet', alpha=alpha, eta0=eta)
    # define evaluation procedure
    cv = RepeatedStratifiedKFold(n_repeats=3, n_splits=10, random_state=1)
    # evaluate the model
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    acc_score = mean(scores)
    return acc_score

def step(currhyperparams, step_size):
    #unpack the current hyper parameters
    curr_eta , curr_alpha = currhyperparams
    #step eta
    new_eta = curr_eta + randn() * step_size
    #check the bounds of eta
    if new_eta < 0.0:
        new_eta = 1e-8 
    if new_eta > 1.0:
        new_eta = 1.0
    #new_eta = (1e-8, new_eta)[if new_eta <= 0]() using ternary operator :p
    #step alpha
    new_alpha = curr_alpha + randn() * step_size
    #check the bounds of alpha
    if new_alpha < 0.0:
        new_alpha = 0.0 
    #returning the new hyperparameters
    return [new_eta, new_alpha]

def hill_climbing(X, y, objfunc, n_iter, step_size):
    '''
    Hill Climbing local search algorithm
    '''
    #starting point for search
    solution_hyperparams = [rand(), rand()]
    # evaluate the initial point
    init_obj_acc_score = obj_func(X, y, solution_hyperparams)
    #run the hill climbing algorithm
    for i in range(n_iter):
        # take a step
        new_candidate_params = step(solution_hyperparams, step_size)
        #evaluate the new candidate
        new_obj_acc_score = obj_func(X, y, new_candidate_params)
        #check if we should keep the new point
        if new_obj_acc_score >= init_obj_acc_score:
            # store the new point
            acc_score, solution_hyperparams = new_obj_acc_score, new_candidate_params
            # report progress
            print(">%d, hyperparameters=%s, score=%.5f" % (i, solution_hyperparams, acc_score))
    return [solution_hyperparams, acc_score]

#define the dataset
X, y = make_classification(n_samples=1000, n_features=5, n_informative=2, n_redundant=1, random_state=1)
#defining the number of iterations
n_iter = 100
#defining the step size/learning rate
step_size = 0.1
#perform the hill climbing
config, mean_score = hill_climbing(X, y, obj_func, n_iter, step_size)
print('Done!')
print(f"The hyperparmeter chosen using hill climbing are {config}"
     f"and the maximum accuracy reported is {mean_score}")
