import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
from random import random 
from scipy.optimize import minimize


def get_param(df,params, param):
    """get set of parameters from the animal fits to use in simulation.
    params_list - column in fits_df with list of parameters names, 
    match with the parameter from params column by index"""
    index = df.params_list[0].index(param)
    return params[index]


class Bandit():
    """for simulations - set reward probabilities and blocks, takes actions and gives rewards"""
    def __init__ (self,p):
        self.probs = p #list of reward probabilities associated with 2 choices
        self.outcomes = [0,1]
        
    def give_reward(self, action):
        reward = int(np.random.choice(self.outcomes,1,p=[(1-self.probs[action]),self.probs[action]]))
        self.reward = reward
    
    def record_optimal_action(self):
        self.optimal_action = np.argmax(self.probs)
    
    def switch(self):
        self.probs = [self.probs[1],self.probs[0]]
        
        
#code to fit bias for random choices
class random_fit(object):
    """choice is random with bias to one side"""
    def __init__(self, df):
        self.n_actions = 2
        self.df = df 
        self.bounds = [(0,1)] #define fit bounds for parameters
        self.guess = np.random.beta(1,1)#define initial guess for parameter fit
        
    def neg_log_likelihood(self, params):
        df = self.df
        bias = params[0]
        
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0
        for action, reward in zip(actions, rewards):
            prob = np.array([bias, (1-bias)])
            prob_log += np.log(prob[action])
        return -prob_log

    def model_estimation(self):
        r = minimize(self.neg_log_likelihood, self.guess,
                     method='Nelder-Mead', bounds = self.bounds)
        return r  
    
    #fit each session for each mouse separately; split conditions for each mouse if available
    def fit_data(self):
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique().tolist():
            for conditions in self.df[self.df.mouse==mice]['condition'].unique().tolist():
                df1 = self.df[(self.df.mouse == mice)&(self.df.condition == conditions)]
                df1 = df1.reset_index(drop = True)
                for sessions in df1.session.unique().tolist(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)
            
                    fit = random_fit(df2)
                    output = fit.model_estimation()
                              
                    BIC = len(output.x)*np.log(len(df2))+2*output.fun
                    AIC = 2*len(output.x) + 2*output.fun

                    temp =pd.DataFrame({'model': 'random_choice','params':[output.x], 
                                        'guess':[self.guess],'params_list': ['bias'], 
                                        'loglik':output.fun, 'BIC':BIC,'AIC':AIC,
                                        'mouse':mice,'session':sessions,'condition':[conditions]})
                    fit_df = pd.concat([fit_df, temp])
        return fit_df
    
    
    
    
class WSLS(object):
     """ repeat rewarded actions and switches away from unrewarded actions with probability 1 - random/2 
     choose the other option (switching after rewards, staying after losses) with probability random/2 """
    def __init__(self,random):
        self.random = random #r

    def get_choice_probs(self,reward, action):
        if reward==1:
            probs = np.ones(2) * self.random/2
            probs[action] = 1 - self.random/2       
        if reward==0:    
            probs = np.ones(2) * (1 - self.random/2)
            probs[action] = self.random/2
        self.probs = probs
        
    def choose_action(self):
        actions = range(2)
        action = np.random.choice(actions, p=self.probs)
        self.action = action
    
    
class WSLS_fit(object):
    def __init__(self, df):
        self.n_actions = 2
        self.df = df
        self.bounds = [(0,1)]
        self.guess = np.random.beta(1,1)
        
    def neg_log_likelihood(self, params):
        df = self.df
        random = params[0]
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0
        for i, (action, reward) in enumerate (zip(actions, rewards)):
            if i ==0:
                prob = np.array([0.5,0.5])
            if i > 0:
                if rewards[i-1]==1:
                    prob = np.ones(2)*random/2
                    prob[actions[i-1]] = 1 - random/2
                if rewards[i-1]==0:    
                    prob = np.ones(2)*(1-random/2)
                    prob[actions[i-1]] = random/2
            prob_log += np.log(prob[action])
        return -prob_log

    def model_estimation(self):
        bounds = (self.bounds)
        r = minimize(self.neg_log_likelihood,self.guess,
                     method='Nelder-Mead',
                     bounds=bounds)
        return r

    def fit_data(self):
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique().tolist():
            for conditions in self.df[self.df.mouse==mice]['condition'].unique().tolist():
                df1 = self.df[(self.df.mouse == mice)&(self.df.condition == conditions)]
                df1 = df1.reset_index(drop = True)
                for sessions in df1.session.unique().tolist(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)
            
                    fit = WSLS_fit(df2)
                    output = fit.model_estimation()
 
                    BIC = len(output.x)*np.log(len(df2))+2*output.fun
                    AIC = 2*len(output.x) + 2*output.fun

                    temp =pd.DataFrame({'model': 'WSLS','params':[output.x], 
                                        'guess':[self.guess],'params_list': ['epsilon'],
                                        'loglik':output.fun, 'BIC':BIC,'AIC':AIC,
                                        'mouse':mice,'session':sessions,'condition':[conditions]})                

                    fit_df = pd.concat([fit_df, temp])
        return fit_df


    
class choice_kernel(object):
    """ compute a choice kernel for each action to keep track of how frequently that option was chosen in the past
    use CK for both actions to make choice according to a softmax function """
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
        self.CK = 0* np.ones(2)

    def get_choice_probs(self):
        num = np.exp(self.CK * self.beta)
        den = np.exp(self.CK * self.beta).sum()
        self.probs = num / den
        
    def choose_action(self):
        actions = range(2)
        action = np.random.choice(actions, p=self.probs)
        self.action = action
        
    def update_choice_kernel(self, action, reward):
        C = np.array([0,0])
        C[action] = 1
        self.CK[action] += self.alpha *(C[action] - self.CK[action])  
        

class CK_fit(object):
    def __init__(self, df):
        self.n_actions = 2
        self.df = df
        self.bounds = [(0,1), (0,5)]
        self.guess = [np.random.beta(1,1), np.random.gamma(3,1)]
        self.params_list = ['alpha_CK','beta_CK']
        
    def neg_log_likelihood(self, params):
        df = self.df
        alpha = params[0]
        beta = params[1]
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0
        CK = 0* np.ones(self.n_actions)

        for action, reward in zip(actions, rewards):
            C = np.array([0,0])
            C[action] = 1
            
            CK[action] += alpha *(C[action] - CK[action])   
            prob_log += np.log(softmax(CK, beta)[action])            
        return -prob_log

    def model_estimation(self):
        bounds = (self.bounds)
        r = minimize(self.neg_log_likelihood, self.guess,
                     method='Powell',
                     bounds=bounds)
        return r

    def fit_data(self):
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique().tolist():
            for conditions in self.df[self.df.mouse==mice]['condition'].unique().tolist():
                df1 = self.df[(self.df.mouse == mice)&(self.df.condition == conditions)]
                df1 = df1.reset_index(drop = True)       
                for sessions in df1.session.unique().tolist(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)
            
                    fit = CK_fit(df2)
                    output = fit.model_estimation()
                              
#                     BIC = np.log(len(df2))+2*output.fun
#                     AIC = 2*len(output.x) + 2*output.fun
                    BIC = len(output.x)*np.log(len(df2))+2*output.fun
                    AIC = 2*len(output.x) + 2*output.fun

                    temp = pd.DataFrame({'model':'choice_kernel', 'params':[output.x], 
                                         'guess':[self.guess],'params_list':[self.params_list], 
                                            'loglik':output.fun, 'BIC':BIC,'AIC':AIC,
                                            'mouse':mice,'session':sessions,'condition':conditions})                

                    fit_df = pd.concat([fit_df, temp])

        return fit_df



class RW_model(object):
    """ on every trial t the expected value Q_t (a) of a chosen action a is updated by the reward prediction error (RPE), 
    the difference between the choice outcome r_t and previous expected value, scaled by the learning rate α;

    The choice probabilities estimated based on the action values according to a softmax function 
    simplest model only has alpha and beta, based on name augmentations (called through sim or fit class) adds extra parameters;
    if no addition - that parameter is set to 0 or 1 so it has no effect on the model """

    def __init__(self, alpha_r, alpha_ur, alpha_CK, 
                 beta, beta_CK, bias, rho, 
                 alpha_r_c,alpha_ur_c, delta, P,
                reset_to_mean= False,reset_to_neutral= False, forget_to_mean= False,forget_to_neutral= False):
     
        self.n_actions = 2
        self.alpha_r = alpha_r #rewarded alpha
        self.alpha_ur = alpha_ur#unrewarded alpha
        self.alpha_CK = alpha_CK #alpha for choice kernel update (for trial repretition in a sticky model)
        self.beta = beta
        self.beta_CK = beta_CK #beta for stickiness part in softmax
        self.bias = bias #side bias
        self.rho = rho #scaling of reward to change reward sensitivity
        self.P = P #another way to do stickiness - increase tendency to repeat previous choice
        
        
        self.delta = delta #forgetting rate
        self.alpha_r_c = alpha_r_c #rewarded alpha for counterfactual updating (unchosen trial)
        self.alpha_ur_c = alpha_ur_c #unrewarded alpha for counterfactual updating (unchosen trial)

        #different ways of defining forgetting - tried different types, forget_to_mean makes most sense? 
        self.reset_to_mean=reset_to_mean; self.reset_to_neutral=reset_to_neutral; self.forget_to_mean=forget_to_mean; self.forget_to_neutral=forget_to_neutral
        
        #difiner initial Q and CK
        self.Q = np.array([0.5,0.5])
        self.CK = 0* np.ones(self.n_actions)
        self.action = 10 #set something other than 0 or 1 for the first trial 
        
    
            
    def update_belief(self, action,reward):
        C = np.array([0,0])#choice tracker 
        
        #pick alpas for chosen and unchosen trial updates based on trial outcome 
        if reward ==1:
            alpha = self.alpha_r
            alpha_c = self.alpha_r_c
        else:
            alpha = self.alpha_ur
            alpha_c = self.alpha_ur_c
            
        #Q update 
        self.Q[action] += alpha * (self.rho*reward - self.Q[action])
        
        #counterfactual update (double updating) 
        if self.alpha_r_c != 0:
            self.Q[1-action] += alpha_c * ((self.rho*reward-1) * (-1) - self.Q[1-action]) #treat reward as opposite on the opposite lever        

        #forgetting of the unchosen option
        #different forgetting options
        if self.delta != 0:
            #reduce value of the option that's not chosen  - restricted to average value 
            if self.forget_to_mean ==True:
                mean_Q = np.mean(self.Q)
                if self.Q[1-action] >= mean_Q: delta = self.delta #positive delta = reduce value
                else: delta = -self.delta #negative delta = increase value to mean
                #if overshoot either side of mean - set at mean
                
                forget = (1-delta) * self.Q[1-action]
                
                if (self.Q[1-action] >= mean_Q and forget < mean_Q) or (self.Q[1-action] <  mean_Q and forget > mean_Q): 
                    self.Q[1-action] = mean_Q
                else: self.Q[1-action] = forget
                
            #forgetting to 0.5 - Qs can go up or down towards 0.5
            elif self.forget_to_neutral ==True:
                if self.Q[1-action] >=0.5: delta = self.delta #reduce value to 0.5
                else: delta = -self.delta #increase value to 0.5
                
                forget = (1-delta) * self.Q[1-action]
                
                #if overshoot either side of 0.5 - set at 0.5
                if (self.Q[1-action] >=0.5 and forget <= 0.5) or (self.Q[1-action] < 0.5 and forget > 0.5): 
                    self.Q[1-action] = 0.5
                else: self.Q[1-action] = forget     
            
            #reduce value of the option that's not chosen = forgetting to 0 
            else: self.Q[1-action] = (1-self.delta) * self.Q[1-action] 
        #resetting - no delta, just immediate reset 
        if self.reset_to_mean== True: self.Q[1-action] = np.mean(self.Q)
        if self.reset_to_neutral == True: self.Q[1-action] = 0.5
        
        C[action] = 1
        self.CK[action] += self.alpha_CK *(C[action] - self.CK[action])   
            
    def get_choice_probs(self):
        V = np.copy(self.Q)
        V[0] = V[0] + self.bias
        if self.action != 10: #not first trial
            V[self.action] = V[self.action] + self.P #on trials 2+ add P to V to promote past choice repetition 
        
        #softmax with values and choice kernels to get choice probabilities for 2 actions 
        num = np.exp(V * self.beta + self.CK * self.beta_CK) 
        den = np.exp(V * self.beta + self.CK * self.beta_CK).sum()
        self.probs = num/den
        
    def choose_action(self):
        actions = range(2)
        action = np.random.choice(actions, p=self.probs)#pick an action based on probs from softmax
        self.action = action
        

class RW_fit(object):
    """ takes data as pandas df, choices 'right_or_left' column - 0 or 1; rewards 'reward' - 0 or 1; 
    and model - name of the model to fit, takes different additions like RP, bias to add correspondign parameters for fitting """
    def __init__(self, df, model, go_bias = False, RP = False, sensitivity = False, forgetting = False,
                 sticky = False, DSA = False, same_alpha = False, scale_alpha = False, repeat = False,
                reset_to_mean=False,reset_to_neutral=False, forget_to_mean=False,forget_to_neutral=False):
        
        self.df = df
        self.model = model
        self.go_bias = go_bias; self.RP = RP; self.sensitivity = sensitivity; self.forgetting = forgetting
        self.sticky = sticky; self.DSA = DSA; self.same_alpha = same_alpha; self.scale_alpha = scale_alpha; self.repeat = repeat
        #extra forgetting options
        self.reset_to_mean=reset_to_mean; self.reset_to_neutral=reset_to_neutral; self.forget_to_mean=forget_to_mean; self.forget_to_neutral=forget_to_neutral

        #use model name to add additional augments to the model
        if 'RP' in model: self.RP = True #reward-punishment = different alphas for rewarded/unrewarded trials 
        if 'DSA' in model: self.DSA = True #double updating - counterfactual updating of values for unchosen option
        if 'same_alpha' in model: self.same_alpha = True #for DSA - whether use same or different alpha for counterfactual update 
        if 'scale_alpha' in model: self.scale_alpha = True #for DSA - can get alpha_c by scaling alpha 
                
        if 'forgetting' in model: self.forgetting = True 
        if 'bias' in model: self.go_bias = True
        if 'sensitivity' in model: self.sensitivity = True #scaling of reward by rho - (0-1)
        if 'sticky' in model: self.sticky = True #addition of a choice kernel 
        if 'repeat' in model: self.repeat = True #increase tendency to repeat the choice 
        
        #forgetting options 
        if 'reset_to_mean' in model: self.reset_to_mean = True #one trial reset of unchosen option to mean
        if 'reset_to_neutral' in model: self.reset_to_neutral = True #one trial reset of unchosen option to 0.5
        if 'forget_to_mean' in model: self.forget_to_mean = True #gradual decay bound at 0.5 
        if 'forget_to_neutral' in model: self.forget_to_neutral = True #gradual decay bound at mean 
            
        #simplest model only has alpha and beta - start with bounds and guesses for those, then add other augments 
        self.bounds = [(0,1),(0,100)]
        self.guess = [np.random.beta(1,1),np.random.gamma(3,1)]
        self.params_len = 2
        self.params_list = ['alpha_r', 'beta']

        #add additional things to the bounds/guess and parameter list according to the different components in the model name 
        if self.RP == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1; self.alpha_ur_index = self.params_len-1
            self.params_list.append('alpha_ur') #keep track of added parameters and their order by adding names to list
            
        if self.DSA == True: 
            if self.scale_alpha== True: #for double updating multiply a_r and a_r by kappa               
                self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
                self.params_len +=1; self.kappa_index = self.params_len-1
                self.params_list.append('kappa')

            elif self.same_alpha == True: pass
            else: #if same_alpha - use same a_r and a_ur for updating of both levers
                self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
                self.params_len +=1; self.alpha_r_c_index = self.params_len-1
                self.params_list.append('alpha_r_c')
                
                if self.RP ==True:
                    self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
                    self.params_len +=1; self.alpha_ur_c_index = self.params_len-1
                    self.params_list.append('alpha_ur_c')
            
        if self.forgetting == True and self.reset_to_neutral ==False and self.reset_to_mean == False: 
            
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1; self.delta_index = self.params_len-1
            self.params_list.append('delta')
            
        if self.go_bias == True: 
            self.bounds.append((-1,1)); self.guess.append(np.random.uniform(low=-1.0, high=1.0))
            self.params_len +=1; self.bias_index = self.params_len -1 
            self.params_list.append('bias')

        if self.repeat == True: 
            self.bounds.append((-100,100)); self.guess.append(np.random.normal(-10, 10, 1)[0])
            self.params_len +=1; self.repeat_index = self.params_len -1 
            self.params_list.append('P')
            

        if self.sensitivity == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1
            self.rho_index = self.params_len -1 
            self.params_list.append('rho')
            
        if self.sticky == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.bounds.append((0,10)); self.guess.append(np.random.gamma(3,1))
        
            self.params_len +=2
            self.alpha_CK_index = self.params_len -2
            self.params_list.append('alpha_CK')

            self.beta_CK_index = self.params_len -1
            self.params_list.append('beta_CK')

            
            
        self.bounds = tuple(self.bounds)

    #the function that estimates log likelihood from model run using animals' choices and rewards 
    #next - minimise its output to get the best parameter estimates 
    def neg_log_likelihood(self, params):
        df = self.df
        
        #pick parameters from the params list 
        #if no addition - that parameter is set to 0 or 1 so it has no effect on the model 
        alpha_r = params[0]
        beta = params[1]
        
        if self.RP == True: alpha_ur = params[self.alpha_ur_index]
        else: alpha_ur = params[0]   
            
        if self.DSA == True: 
            if self.scale_alpha== True: 
                alpha_r_c = alpha_r * params[self.kappa_index]
                
                if self.RP == True:  
                    alpha_ur_c = alpha_ur * params[self.kappa_index]
                else: alpha_ur_c = alpha_r_c
                    
            elif self.same_alpha == True: 
                alpha_r_c = alpha_r
                if self.RP == True: alpha_ur_c = alpha_ur
                else: alpha_ur_c = alpha_r_c
                    
            else:
                alpha_r_c = params[self.alpha_r_c_index]
                if self.RP == True:alpha_ur_c = params[self.alpha_ur_c_index]
                else: alpha_ur_c = alpha_r_c         
                
        else: 
            alpha_r_c = 0
            alpha_ur_c = 0
            
        if self.go_bias == True: bias = params[self.bias_index]
        else: bias = 0
            
        if self.repeat == True: P = params[self.repeat_index]
        else: P = 0

        if self.sensitivity == True: rho = params[self.rho_index]
        else: rho =  1
            
        if self.forgetting == True and self.reset_to_neutral ==False and self.reset_to_mean == False: 
            delta = params[self.delta_index]
        else: delta =  0
        
        if self.sticky == True: alpha_CK = params[self.alpha_CK_index]; beta_CK = params[self.beta_CK_index]
        else: alpha_CK =  0; beta_CK = 0;
            
        #take animals choices and trial outcomes
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0
        
        #set up a model with all the parameters
        model = RW_model(alpha_r, alpha_ur, alpha_CK,beta, beta_CK, bias, rho,
                         alpha_r_c,alpha_ur_c, delta, P,
                        reset_to_mean=self.reset_to_mean, reset_to_neutral=self.reset_to_neutral, 
                         forget_to_mean=self.forget_to_mean,forget_to_neutral=self.forget_to_neutral)
        
        #estimate log likelihoods from the model for each trial based on choices/outcomes
        for action, reward in zip(actions, rewards):
            
            model.get_choice_probs()
            prob_log += np.log(model.probs[action])
            model.update_belief(action,reward)    
            
        return -prob_log
    
    
    #minimisation function of neg_log_likelihood, using guesses and bounds composed at initialisation 
    def model_estimation(self):
        r = minimize(self.neg_log_likelihood, self.guess,
                     method='Powell', 
                     bounds=self.bounds)
        return r                    
    
    #take individual mice and sessions from the df to run fits on and set up a fit function    
    def fit_data(self):
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique().tolist():
            for conditions in self.df['condition'].unique():
                df1 = self.df[(self.df['mouse'] == mice)&(self.df['condition'] == conditions)]
                df1 = df1.reset_index(drop = True)
                for sessions in df1.session.unique().tolist(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)

                    fit = RW_fit(df2,model = self.model, go_bias = self.go_bias, RP = self.RP,
                                 sensitivity = self.sensitivity, sticky = self.sticky, DSA = self.DSA,
                                 forgetting = self.forgetting,
                                same_alpha = self.same_alpha, scale_alpha = self.scale_alpha, repeat = self.repeat, reset_to_mean=self.reset_to_mean, reset_to_neutral=self.reset_to_neutral, 
                         forget_to_mean=self.forget_to_mean,forget_to_neutral=self.forget_to_neutral)
                    self.output = fit.model_estimation()
                    
                    BIC = len(self.output.x)*np.log(len(df2))+2*self.output.fun
                    AIC = 2*len(self.output.x) + 2*self.output.fun
                    
                    
                    #output dataframe with best output of model_estimation function per session
                    temp = pd.DataFrame({'model':self.model,'params':[self.output.x], 
                                         'guess':[self.guess],'params_list': [self.params_list],
                                         'loglik':self.output.fun,'BIC':BIC,'AIC':AIC,
                                         'mouse':mice,'session':sessions,'condition':conditions})
                    fit_df = pd.concat([fit_df, temp])
        return fit_df 
    
      
class PH_model(object):
""" Dynamic value updating models (Pearce-Hall) - contain an associability parameter that modulates the learning rate as a function of the absolute magnitude of past RPEs
The κ parameter modulates the action value updating and is equivalent to the learning rate parameter in the basic Q-learning models. 
On the first trial, a_t is a free parameter. The γ parameter controls the temporal dynamics of associability over time """
    def __init__(self, alpha_0, beta,kappa_r,kappa_ur, 
                 gamma,delta,alpha_CK, beta_CK, bias,P):
        self.n_actions = 2
        
        self.alpha_0 = alpha_0
        
        self.beta = beta
        
        self.kappa_r = kappa_r
        self.kappa_ur = kappa_ur
        
        self.gamma = gamma
        
        self.alpha_CK = alpha_CK
        self.beta_CK = beta_CK

        self.bias = bias
        self.delta = delta
        self.P = P
        
        self.Q = np.array([0.5,0.5])
        self.CK = 0* np.ones(self.n_actions)
                
        self.alpha = self.alpha_0
        self.action = 10
        
    def update_belief(self, action,reward):
        C = np.array([0,0])        
        beta = self.beta
        
        RPE = (reward -  self.Q[action])
        
        if self.kappa_r == self.kappa_ur:
            self.Q[action] += self.kappa_r* self.alpha * RPE  
            self.alpha = (1 - self.gamma) * self.alpha + self.gamma * np.abs(RPE)
        
        else:
            if reward ==1: 
                self.Q[action] += self.kappa_r* self.alpha * RPE  
                self.alpha = (1 - self.gamma) * self.alpha + self.gamma * np.abs(RPE)
            else:
                self.Q[action] += self.kappa_ur* self.alpha * RPE  
                self.alpha = (1 - self.gamma) * self.alpha + self.gamma * np.abs(RPE)
        
        if self.delta != 0:
            self.Q[1-action] = (1-self.delta) * self.Q[1-action] #reduce value of the option that's not chosen 
        
        C[action] = 1
        self.CK[action] += self.alpha_CK *(C[action] - self.CK[action])   
            
    def get_choice_probs(self):
        V = np.copy(self.Q)
        V[0] = V[0] + self.bias
        if self.action != 10: #not first trial
            V[self.action] = V[self.action] + self.P
        num = np.exp(V * self.beta + self.CK * self.beta_CK)
        den = np.exp(V * self.beta + self.CK * self.beta_CK).sum()
        self.probs = num/den
        
    def choose_action(self):
        actions = range(2)
        action = np.random.choice(actions, p=self.probs)
        self.action = action
                

class PH_fit(object):
    def __init__(self, df, model, go_bias = False, RP = False, RP_gamma = False, RP_kappa = False, forgetting = False,
                 sticky = False, repeat = False):
        self.df = df
        
        self.model = model
        self.RP = RP; self.go_bias = go_bias; 
        self.forgetting = forgetting; self.sticky = sticky; self.repeat = repeat
        
        if 'RP' in model: self.RP = True 
        if 'forgetting' in model: self.forgetting = True
        if 'bias' in model: self.go_bias = True
        if 'sticky' in model: self.sticky = True
        if 'repeat' in model: self.repeat = True
        
            
        self.bounds = [(0,1),(0,1), (0,1),(0,100)]
        self.params_len = 4
        self.guess = [np.random.beta(1,1),np.random.beta(1,1),np.random.beta(1,1),np.random.gamma(3,1)]
        self.params_list = ['alpha_0', 'kappa_r','gamma', 'beta']

        if self.RP == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1; self.kappa_ur_index = self.params_len-1
            self.params_list.append('kappa_ur')
            
        if self.forgetting == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1; self.delta_index = self.params_len-1
            self.params_list.append('delta')
            
        if self.go_bias == True: 
            self.bounds.append((-1,1)); self.guess.append(np.random.uniform(low=-1.0, high=1.0))
            self.params_len +=1; self.bias_index = self.params_len -1 
            self.params_list.append('bias')

        if self.repeat == True: 
            self.bounds.append((-100,100)); self.guess.append(np.random.normal(-10, 10, 1)[0])
            self.params_len +=1; self.repeat_index = self.params_len -1 
            self.params_list.append('P')
            
        if self.sticky == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.bounds.append((0,10)); self.guess.append(np.random.gamma(3,1))
        
            self.params_len +=2
            self.alpha_CK_index = self.params_len -2
            self.params_list.append('alpha_CK')

            self.beta_CK_index = self.params_len -1
            self.params_list.append('beta_CK')            
            
        self.bounds = tuple(self.bounds)

    def neg_log_likelihood(self, params):
        df = self.df
        alpha_0 = params[0]
        kappa_r = params[1]
        gamma = params[2]
        beta = params[3]
        
        if self.RP == True: kappa_ur = params[self.kappa_ur_index]; 
        else: kappa_ur = params[1]
        
        if self.go_bias == True: bias = params[self.bias_index]
        else: bias = 0
        
        if self.repeat == True: P = params[self.repeat_index]
        else: P = 0

        if self.forgetting == True: delta = params[self.delta_index]
        else: delta =  0
        
        if self.sticky == True: alpha_CK = params[self.alpha_CK_index]; beta_CK = params[self.beta_CK_index]
        else: alpha_CK =  0; beta_CK = 0;
        
        
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0
        
        model = PH_model(alpha_0, beta,kappa_r,kappa_ur, 
                 gamma,delta, alpha_CK, beta_CK, bias,P=P)
        
        for action, reward in zip(actions, rewards):
            model.get_choice_probs()
            prob_log += np.log(model.probs[action])
            model.update_belief(action,reward)    
            
        return -prob_log

    def model_estimation(self):
        r = minimize(self.neg_log_likelihood, self.guess,
                     method='Powell', 
                     bounds=self.bounds)
        return r                     
    
    def fit_data(self):
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique().tolist():
            for conditions in self.df['condition'].unique():
                df1 = self.df[(self.df['mouse'] == mice)&(self.df['condition'] == conditions)]
                df1 = df1.reset_index(drop = True)
                for sessions in df1.session.unique().tolist(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)

                    fit = PH_fit(df2, model = self.model,go_bias = self.go_bias, 
                                 RP = self.RP, sticky = self.sticky, repeat = self.repeat)
                    self.output = fit.model_estimation()             
                    
                    BIC = len(self.output.x)*np.log(len(df2))+2*self.output.fun
                    AIC = 2*len(self.output.x) + 2*self.output.fun
                    
                    temp = pd.DataFrame({'model':self.model,'params':[self.output.x], 
                                         'guess':[self.guess],'params_list': [self.params_list],
                                         'loglik':self.output.fun,'BIC':BIC,'AIC':AIC,
                                         'mouse':mice,'session':sessions,'condition':conditions})
                    fit_df = pd.concat([fit_df, temp])
        return fit_df 
    

class HMM_model(object):
""" State inference models use Bayesian inference and assume that on each trial mice chose their actions based on their belief about the underlying state of the task 
given the history of observed outcomes; the current belief about the state of the task is mapped into action probabilities via a softmax function as in Q-learning models """
    def __init__(self, alpha_r,alpha_ur,alpha_CK, gamma, beta, beta_CK, bias,P):
        #extra augments like in RW models above 
        self.alpha_r = alpha_r #c in the paper - term used for pos estimate after reward 
        self.alpha_ur = alpha_ur #d in the paper - term used for pos estimate after reward omission
        self.alpha_CK = alpha_CK #choice kernel update alpha
        self.gamma = gamma #transition matrix term
        self.beta = beta 
        self.beta_CK = beta_CK
        self.bias = bias #add side bias 
        self.P = P #add choice repetition 
        self.n = 2
        
        self.n = 2
        self.belief_l = 0.5
        
        self.CK = 0* np.ones(2)
        self.action = 10
        
        #transition matrix;  t0=0; t1=1; ts=1
        self.t0=.5    
        self.t1=.5 
        self.ts=.5
        self.Ts   = np.array([[self.t0+self.ts*self.gamma, self.t1-self.ts*self.gamma],[ self.t1-self.ts*self.gamma, self.t0+self.ts*self.gamma]])
        
        p0 = np.array([.5, .5]) #initial belief
        self.pss = p0
        self.pos = np.array([0.5,0.5])

    def get_choice_probs(self):
        V = np.copy(self.pss)
        V[0] = V[0] + self.bias
        
        if self.action != 10: #not first trial
            V[self.action] = V[self.action] + self.P
            
        #softmax with values and choice kernels to get choice probabilities for 2 actions             
        num = np.exp(V * self.beta + self.CK * self.beta_CK)
        den = np.exp(V * self.beta + self.CK * self.beta_CK).sum()
        self.probs = num/den

    def choose_action(self):
        actions = range(self.n)
        action = np.random.choice(actions, p=self.probs)#pick an action based on probs from softmax
        self.action = action

    def update_belief(self, action, reward):
        C = np.array([0,0])
        if reward==1:
            #pos = prob(ot|st) - observation given state; 
            #if action == belief & rewarded  = evidence to support current belief
            #if action != belief & rewarded  = evidence against current belief
            self.pos[action]    = self.t0+self.ts*self.alpha_r
            self.pos[1-action]  = self.t1-self.ts*self.alpha_r
        
        else:
            #if action == belief & non rewarded  = evidence against current belief
            #if action != belief & non rewarded  = evidence to support current belief
            self.pos[action]    = self.t1-self.ts*self.alpha_ur
            self.pos[1-action]  = self.t0+self.ts*self.alpha_ur
            
        C[action] = 1
        
        #update choice kernel 
        self.CK[action] += self.alpha_CK *(C[action] - self.CK[action])   
        
        ps = self.pss * self.pos  #ps = prob(st|ot);pss = prob(st|ot-1); pos = prob(ot|st);  p0 is pss for the first trial; 
        zps = sum(ps) #zps = sum of probs associated with both actions 
        ps = ps/zps #p(s_t|o^t)
        
        self.pss = np.matmul(ps,self.Ts)
        
        
class HMM_fit(object):
    
    def __init__(self, df, model, RP = False, bias = False, sticky = False, 
                 fixed_beta = False, fixed_gamma = False,fixed_aur = False, repeat = False):
        
        self.model = model
        self.df = df
        self.RP = RP; self.bias = bias; self.sticky = sticky; 
        self.fixed_beta = fixed_beta; self.fixed_gamma = fixed_gamma; self.fixed_aur = fixed_aur; self.repeat = repeat

        #use model name to add additional augments to the model
        if 'RP' in model: self.RP = True #reward-punishment = different alphas for rewarded/unrewarded trials 
        if 'bias' in model: self.bias = True
        if 'sticky' in model: self.sticky = True  #addition of a choice kernel 
        if 'fixed_beta' in model: self.fixed_beta = True  #version with fixed beta as suggested in quentin's paper 
        if 'fixed_gamma' in model: self.fixed_gamma = True #for the version with fixed transition matrix 
        if 'fixed_aur' in model: self.fixed_aur = True #in previous fits alpha_ur (d) often came out at the lowest value close to 0 - this version fixes alpha_ur at 0 from the start - so no updating at reward omission
        if 'repeat' in model: self.repeat = True #increase tendency to repeat the choice 

        self.bounds = [(0,1)]
        self.guess = [np.random.beta(1,1)]
        self.params_len = 1
        self.params_list = ['alpha_r']
    
        #add additional things to the bounds/guess and parameter list according to the different components in the model name 
        if self.RP == True:
            if self.fixed_aur == False:
                self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
                self.params_len +=1; self.alpha_ur_index = self.params_len-1
                self.params_list.append('alpha_ur')
        
                
        if self.bias == True: 
            self.bounds.append((-1,1)); self.guess.append(np.random.uniform(low=-1.0, high=1.0))
            self.params_len +=1; self.bias_index = self.params_len -1 
            self.params_list.append('bias')
#             self.model = self.model + '_bias'
        if self.repeat == True: 
            self.bounds.append((-100,100)); self.guess.append(np.random.normal(-10, 10, 1)[0])
            self.params_len +=1; self.repeat_index = self.params_len -1 
            self.params_list.append('P')
            
        
        if self.sticky == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.bounds.append((0,10)); self.guess.append(np.random.gamma(3,1))
        
            self.params_len +=2
            self.alpha_CK_index = self.params_len -2
            self.params_list.append('alpha_CK')

            self.beta_CK_index = self.params_len -1
            self.params_list.append('beta_CK')
   
        if self.fixed_gamma == False:
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1; self.gamma_index = self.params_len-1
            self.params_list.append('gamma')
        
        if self.fixed_beta == False: 
            self.bounds.append((0,100)); self.guess.append(np.random.gamma(3,1))
            self.params_len +=1; self.beta_index = self.params_len-1
            self.params_list.append('beta')

            
    #the function that estimates log likelihood from model run using animals' choices and rewards 
    #next - minimise its output to get the best parameter estimates (params)
    def neg_log_likelihood(self, params):
        df = self.df
        alpha_r = params[0]
        #pick parameters from the params list 
        #if no addition - that parameter is set to 0 or 1 so it has no effect on the model
   
        #option to fix beta or gamma 
        if self.fixed_beta == True:
            beta = 10
        else: beta = params[self.beta_index]
            
        if self.fixed_gamma == True:
            gamma = 0.9
        else: gamma = params[self.gamma_index]
        
        if self.RP == True: 
            if self.fixed_aur == True: alpha_ur = 0
            else: alpha_ur = params[self.alpha_ur_index]   
        else: alpha_ur = alpha_r
                
        if self.bias == True: bias = params[self.bias_index]
        else: bias = 0
            
        if self.repeat == True: P = params[self.repeat_index]
        else: P = 0

        if self.sticky == True: alpha_CK = params[self.alpha_CK_index]; beta_CK = params[self.beta_CK_index]
        else: alpha_CK =  0; beta_CK = 0;
            
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0

        #set up a model with all the parameters
        model = HMM_model(alpha_r,alpha_ur,alpha_CK, gamma, beta, beta_CK, bias,P=P)
        
        for action, reward in zip(actions, rewards):
            model.get_choice_probs()
            prob_log += np.log(model.probs[action])
            model.update_belief(action,reward)     
        return -prob_log   
    
    #minimisation of neg_log_likelihood, using guesses and bounds composed at initialisation 
    def model_estimation(self):
        r = minimize(self.neg_log_likelihood, self.guess,
                     method='Powell',
                     bounds=self.bounds)
        return r
    
    #take individual mice and sessions from the df to run fits on and set up a fit function
    def fit_data(self):
        if self.fixed_beta == True: self.params_list.append('beta')
        if self.fixed_gamma == True: self.params_list.append('gamma')
        if self.fixed_aur == True: self.params_list.append('alpha_ur')
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique():
            for conditions in self.df['condition'].unique():
                df1 = self.df[(self.df['mouse'] == mice)&(self.df['condition'] == conditions)]
                df1 = df1.reset_index(drop = True)
                for sessions in df1.session.unique(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)

                    fit = HMM_fit(df2,model = self.model, RP = self.RP, bias = self.bias, sticky = self.sticky, 
                                  fixed_beta = self.fixed_beta,fixed_gamma = self.fixed_gamma, repeat = self.repeat)
                    output = fit.model_estimation()#call model estimation; 
                    #output = result of minimisation- output.fun = lowest loglik, output.x = parameters
#                     
                    BIC = len(output.x)*np.log(len(df2))+2*output.fun
                    AIC = 2*len(output.x) + 2*output.fun
        
                    params = [i for i in output.x]
                    if self.fixed_beta == True: params.append(10);
                    if self.fixed_gamma == True: params.append(0.9);
                    if self.fixed_aur == True: params.append(0);
                    temp = pd.DataFrame({'model':self.model,'params':[params], 
                                         'guess':[self.guess],'params_list': [self.params_list],
                                         'loglik':output.fun,'BIC':BIC,'AIC':AIC,
                                         'mouse':mice,'session':sessions,'condition':conditions})
                    fit_df = pd.concat([fit_df, temp])

        return fit_df                      


class IO_model(object):
    """  Ideal observer - similar to HMM but takes known reward probabilities into reward matrix; 
    didn't use so didn't check as thoroughly, but seems ok? """ 
    def __init__(self, beta, gamma, alpha_CK, beta_CK, bias, P, rew_exp = False):
        
        #parameters
        self.beta =beta; self.gamma = gamma; self.P = P;
        self.alpha_CK = alpha_CK; self.beta_CK = beta_CK; self.bias = bias
        
        #model conditions
        self.rew_exp = rew_exp;
        self.Ts   = np.array([[0.5 + 0.5 * gamma, 0.5 - 0.5 * gamma],[0.5 - 0.5 * gamma, 0.5 + 0.5 * gamma]])
        
        self.Rm = np.array([[0.7, 0.1], [0.1, 0.7]]) 
        
        self.n = 2
        p0 = np.array([.5, .5]) #initial belief
        self.pss = p0
        
        self.CK = 0* np.ones(2)
        self.action = 10

    def get_choice_probs(self):
        if self.rew_exp == True: 
            self.rew_prob = np.matmul(self.pss, self.Rm)  #expectation of reward = rew probs in 2 states scaled by probability of states
            V = np.copy(self.rew_prob)
            
        else: V = np.copy(self.pss) #or use probablity of state to make choice 
        
        V[0] = V[0] + self.bias
        if self.action != 10: #not first trial
            V[self.action] = V[self.action] + self.P
        num = np.exp(V * self.beta + self.CK * self.beta_CK)
        den = np.exp(V * self.beta + self.CK * self.beta_CK).sum()
        self.probs = num/den

    def choose_action(self):
        actions = range(self.n)
        action = np.random.choice(actions, p=self.probs)
        self.action = action
        
    def update_belief(self, action, reward):
        C = np.array([0,0])
        C[action] = 1
        self.CK[action] += self.alpha_CK *(C[action] - self.CK[action])   
        
#         #combination of hmm and ideal observer 
        #pick the column of the Rm according to the taken action
        pos = (self.Rm[:][action]**reward)*((1-self.Rm[:][action])**(1-reward))  
        ps = pos *  self.pss #ps = prob(st|ot);pss = prob(st|ot-1); pos = prob(ot|st);  p0 is pss for the first trial; this is correct 
        zps = sum(ps) #zps = sum of probs associated with both actions 
        ps = ps/zps #p(s_t|o^t)
        self.pss = np.matmul(ps,self.Ts)

class IO_fit(object):
    
    def __init__(self, df, model,fixed_beta = False,fixed_gamma = False,
                 rew_exp = False, bias = False, sticky = False, repeat = False):
        self.df = df
        
        #model conditions
        self.rew_exp = rew_exp; self.bias = bias; self.sticky = sticky; self.fixed_gamma = fixed_gamma; self.repeat = repeat
        
        if 'rew_exp' in model: self.rew_exp = True
        if 'bias' in model: self.bias = True
        if 'sticky' in model: self.sticky = True
#         if 'fixed_beta' in model: self.fixed_beta = True
        if 'fixed_gamma' in model: self.fixed_gamma = True
        if 'repeat' in model: self.repeat = True
        
        self.bounds = [(0,100)]
        self.guess = [np.random.gamma(3,1)]
        self.params_len = 1
        self.params_list = ['beta']
        self.model = model

#         if self.rew_exp ==True: self.model = self.model + '_rew_exp'
#         self.bounds = tuple(self.bounds)
        
        if self.bias == True: 
            self.bounds.append((-1,1)); self.guess.append(np.random.uniform(low=-1.0, high=1.0))
            self.params_len +=1; self.bias_index = self.params_len -1 
            self.params_list.append('bias')
#             self.model = self.model + '_bias'
        
        if self.repeat == True: 
            self.bounds.append((-100,100)); self.guess.append(np.random.normal(-10, 10, 1)[0])
            self.params_len +=1; self.repeat_index = self.params_len -1 
            self.params_list.append('P')
        
        if self.sticky == True: 
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.bounds.append((0,10)); self.guess.append(np.random.gamma(3,1))
        
            self.params_len +=2
            self.alpha_CK_index = self.params_len -2
            self.params_list.append('alpha_CK')

            self.beta_CK_index = self.params_len -1
            self.params_list.append('beta_CK')

#             self.model = self.model + '_sticky'    
        
        if self.fixed_gamma == False:
            self.bounds.append((0,1)); self.guess.append(np.random.beta(1,1))
            self.params_len +=1; self.gamma_index = self.params_len-1
        self.params_list.append('gamma')
    
    def neg_log_likelihood(self, params):
        df = self.df
        beta = params[0]
           
        if self.fixed_gamma == True:
            gamma = 0.9
        else: gamma = params[self.gamma_index]
            
        
        if self.bias == True: bias = params[self.bias_index]
        else: bias = 0
            
        if self.repeat == True: P = params[self.repeat_index]
        else: P = 0

        if self.sticky == True: alpha_CK = params[self.alpha_CK_index]; beta_CK = params[self.beta_CK_index]
        else: alpha_CK =  0; beta_CK = 0;
        
            
        actions, rewards = df['right_or_left'].values.astype(int), df['reward'].values.astype(int)
        prob_log = 0

        model = IO_model(beta = beta, gamma = gamma, alpha_CK = alpha_CK, 
                         beta_CK = beta_CK, bias = bias,P=P,rew_exp = self.rew_exp)

        for action, reward in zip(actions, rewards):
            model.get_choice_probs()
            prob_log += np.log(model.probs[action])
            model.update_belief(action,reward)     
        return -prob_log          

    def model_estimation(self):
        r = minimize(self.neg_log_likelihood, self.guess,
                     method='Powell',
                     bounds=self.bounds)
        return r
    
    def fit_data(self):
        fit_df = pd.DataFrame()
        for mice in self.df['mouse'].unique():
            for conditions in self.df['condition'].unique():
                df1 = self.df[(self.df['mouse'] == mice)&(self.df['condition'] == conditions)]
                df1 = df1.reset_index(drop = True)
                for sessions in df1.session.unique(): 
                    df2 = df1[df1.session==sessions]
                    df2 = df2.reset_index(drop = True)

                    fit = IO_fit(df2,model = self.model,rew_exp = self.rew_exp, 
                                fixed_gamma = self.fixed_gamma, repeat = self.repeat)
                    output = fit.model_estimation()
                    
                    BIC = len(output.x)*np.log(len(df2))+2*output.fun
                    AIC = 2*len(output.x) + 2*output.fun
        
                    params = [i for i in output.x]
            
                    if self.fixed_gamma == True: params.append(0.9)
                    temp = pd.DataFrame({'model':self.model, 'params':[params], 
                                         'guess':[self.guess],'params_list': [self.params_list],
                                         'loglik':output.fun,'BIC':BIC,'AIC':AIC,
                                         'mouse':mice,'session':sessions,'condition':conditions})
                    fit_df = pd.concat([fit_df, temp])
        return fit_df   
    




#simulation classes

class random_sim(object):
    """ simulation of random choices; if set from_fit True, can use parameters from fit_df fits to data,
    can set number of trials in each run and number of simulation runs """  
    def __init__(self,fit_df = 'NA',from_fit = False):
        self.from_fit = from_fit
        if from_fit==True: self.fit_df = fit_df[fit_df.model=='random_choice'].reset_index(drop = True)
            
        self.model = 'random_choice'
        self.params_list = ['bias']
        

    def simulate(self, runs =300):
        sim_df = pd.DataFrame()
        
        if self.from_fit==True: runs = len(self.fit_df)
            
        for run in range(runs):
            num_trials = 300
            self.params = []
            
            if self.from_fit==True: 
                params = self.fit_df.loc[run,'params']
                mouse_session = self.fit_df.loc[run,'mouse'] + '_' + str(self.fit_df.loc[run,'session'])
                mouse_model = self.fit_df.loc[run,'mouse'] + '_' + self.model
                
                bias = get_param(self.fit_df,params,'bias')
                
            else: bias = np.random.beta(1,1)
            self.params.append(bias)

            bandit = Bandit([0.7,0.1])
            agent = random_choice(bias)
            
            switch = int(np.random.choice(range(10,32),1))
                   
            actions = np.zeros(num_trials)
            optimal = np.zeros(num_trials)
            rewards = np.zeros(num_trials)
            correct_choices = np.zeros(num_trials) 
            track_correct = 0

            for t in range(num_trials):
                agent.get_choice_probs()
                agent.choose_action() #choose lever
                
                actions[t] = agent.action #record choice
                bandit.record_optimal_action() #record optimal choice
                optimal[t] = bandit.optimal_action*(-2)+1
                bandit.give_reward(agent.action) #give reward
                rewards[t] = bandit.reward #record reward
                correct_choices[t] = (agent.action==bandit.optimal_action)*1
                track_correct += (agent.action==bandit.optimal_action)*1

                if track_correct == switch:
                    bandit.switch()
                    switch = int(np.random.choice(range(10,32),1))
                    track_correct = 0


            if self.from_fit: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials,
                                    'mouse_session': mouse_session, 'mouse_model':mouse_model
                                                       })
            else: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials
                                                       })

        
            sim_df = pd.concat([sim_df, temp])
            sim_df = sim_df.reset_index(drop=True)
            
        return(sim_df)



class WSLS_sim(object):
    """ simulation of WSLS; if set from_fit True, can use parameters from fit_df """  
    def __init__(self,fit_df = 'NA',from_fit = False):
        self.from_fit = from_fit
        if from_fit==True: 
            self.fit_df = fit_df[fit_df.model=='WSLS'].reset_index(drop = True)
        self.model = 'WSLS'
        self.params_list = ['epsilon']
    
    def simulate(self, runs =100):
        sim_df = pd.DataFrame()
        if self.from_fit==True: runs = len(self.fit_df)
            
        for run in range(runs):
            num_trials = 300
            self.params = []
            
            if self.from_fit==True: 
                params = self.fit_df.loc[run,'params']
                mouse_session = self.fit_df.loc[run,'mouse'] + '_' + str(self.fit_df.loc[run,'session'])
                mouse_model = self.fit_df.loc[run,'mouse'] + '_' + self.model
                
                epsilon = get_param(self.fit_df,params,'epsilon')
                
            else: epsilon = np.random.beta(1,1)
            self.params.append(epsilon)
                    
            bandit = Bandit([0.7,0.1])
            agent = WSLS(epsilon)
            
            switch = int(np.random.choice(range(10,32),1))
                   
            actions = np.zeros(num_trials)
            rewards = np.zeros(num_trials)
            optimal = np.zeros(num_trials)
            correct_choices = np.zeros(num_trials) 
            track_correct = 0

            for t in range(num_trials):
                if t ==0: action = np.random.choice([0,1], p=[0.5,0.5]);reward = np.random.choice([0,1], p=[0.5,0.5])
                agent.get_choice_probs(reward,action)
                agent.choose_action() #choose lever
                
                action = agent.action
                actions[t] = agent.action #record choice
                bandit.record_optimal_action() #record optimal choice
                optimal[t] = bandit.optimal_action*(-2)+1
                
                bandit.give_reward(agent.action) #give reward
                rewards[t] = bandit.reward #record reward
                reward = bandit.reward
                
                correct_choices[t] = (agent.action==bandit.optimal_action)*1
                track_correct += (agent.action==bandit.optimal_action)*1

                if track_correct == switch:
                    bandit.switch()
                    switch = int(np.random.choice(range(10,32),1))
                    track_correct = 0


            if self.from_fit: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials,
                                    'mouse_session': mouse_session, 'mouse_model':mouse_model
                                                       })
            else: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials
                                                       })
            sim_df = pd.concat([sim_df, temp])
            sim_df = sim_df.reset_index(drop=True)
            
        return(sim_df)
    
class CK_sim(object):
    def __init__(self,fit_df = 'NA',from_fit = False):
        self.from_fit = from_fit
        if from_fit==True: 
            self.fit_df = fit_df[fit_df.model=='choice_kernel'].reset_index(drop = True)
        self.model = 'choice_kernel'
        self.params_list = ['alpha_CK','beta_CK']
    
    def simulate(self, runs = 100):
        sim_df = pd.DataFrame()
        if self.from_fit==True: runs = len(self.fit_df)
            
        for run in range(runs):
            num_trials = 300
            self.params = []
            
            if self.from_fit==True: 
                params = self.fit_df.loc[run,'params']
                mouse_session = self.fit_df.loc[run,'mouse'] + '_' + str(self.fit_df.loc[run,'session'])
                mouse_model = self.fit_df.loc[run,'mouse'] + '_' + self.model

                alpha = get_param(self.fit_df,params,'alpha_CK')
                beta = get_param(self.fit_df,params,'beta_CK')
                
            else: alpha = np.random.beta(1,1); beta = np.random.gamma(3,1); #beta in fits is often maxed out at 5 
            self.params.append(alpha)
            self.params.append(beta)
                    
            bandit = Bandit([0.7,0.1])
            agent = choice_kernel(alpha,beta)
            
            switch = int(np.random.choice(range(10,32),1))
                   
            actions = np.zeros(num_trials)
            rewards = np.zeros(num_trials)
            optimal = np.zeros(num_trials)
            correct_choices = np.zeros(num_trials) 
            track_correct = 0

            for t in range(num_trials):
                agent.get_choice_probs()
                
                agent.choose_action() #choose lever
                actions[t] = agent.action #record choice

                bandit.give_reward(agent.action) #give reward
                rewards[t] = bandit.reward #record reward

                agent.update_choice_kernel(agent.action,bandit.reward) #updatae CK                          

                bandit.record_optimal_action()
                optimal[t] = bandit.optimal_action*(-2)+1

                correct_choices[t] = (agent.action==bandit.optimal_action)*1
                track_correct += (agent.action==bandit.optimal_action)*1
                if track_correct == switch:
                    bandit.switch()
                    switch = int(np.random.choice(range(10,32),1))
                    track_correct = 0


            if self.from_fit: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials,
                                    'mouse_session': mouse_session, 'mouse_model':mouse_model
                                                       })
            else: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials
                                                       })
            sim_df = pd.concat([sim_df, temp])
            sim_df = sim_df.reset_index(drop=True)
            
        return(sim_df)    

                         
class RW_sim(object):
    """ simulation of RW strategy; if set from_fit True, can use parameters from fit_df fits to data,
    can set number of trials in each run and number of simulation runs;
    similar to fits - takes model name and adds augmentations based on the name;
    setting track_var = True keeps track of Qs and RPEs """  
    def __init__(self,  model = 'RW', fit_df = 'NA',from_fit = False, 
                 go_bias = False, RP = False, sensitivity = False, forgetting = False,
                 sticky = False, DSA = False, same_alpha = False, scale_alpha = False, repeat = False,
                 reset_to_mean=False,reset_to_neutral=False, forget_to_mean=False,forget_to_neutral=False):
        
        self.Q = 0.5* np.ones(2)
        self.from_fit = from_fit
        if from_fit==True: #option to use 
            self.fit_df = fit_df.reset_index(drop = True)
            model = self.fit_df.loc[0,'model']
            
        self.go_bias = go_bias; self.RP = RP; self.sensitivity = sensitivity; self.forgetting = forgetting,
        self.sticky = sticky; self.DSA = DSA; self.same_alpha = same_alpha; self.scale_alpha = scale_alpha; self.repeat = repeat;
        #extra forgetting options
        self.reset_to_mean=reset_to_mean; self.reset_to_neutral=reset_to_neutral; self.forget_to_mean=forget_to_mean; self.forget_to_neutral=forget_to_neutral
                         
                
        self.params_len = 2
        self.params_list = ['alpha_r', 'beta']
        self.model = model
        
        if 'RP' in model: self.RP = True
        if 'DSA' in model: self.DSA = True
        if 'same_alpha' in model: self.same_alpha = True
        if 'scale_alpha' in model: self.scale_alpha = True
                
        if 'forgetting' in model: self.forgetting = True
        if 'bias' in model: self.go_bias = True
        if 'sensitivity' in model: self.sensitivity = True
        if 'sticky' in model: self.sticky = True
        if 'repeat' in model: self.repeat = True
                         
        #forgetting options 
        if 'reset_to_mean' in model: self.reset_to_mean = True #one trial reset of unchosen option to mean
        if 'reset_to_neutral' in model: self.reset_to_neutral = True #one trial reset of unchosen option to 0.5
        if 'forget_to_mean' in model: self.forget_to_mean = True #gradual decay bound at 0.5 
        if 'forget_to_neutral' in model: self.forget_to_neutral = True #gradual decay bound at mean 
        
        
        #self.model = 'RW'
        if self.RP == True: self.params_list.append('alpha_ur'); #self.model = self.model + '_RP'

                
        if self.DSA == True: 
#             self.model = self.model + '_DSA'
            if self.scale_alpha== True: #for double updating multiply a_r and a_r by kappa               
                self.params_list.append('kappa')
#               self.model = self.model + '_scale_alpha'
            elif self.same_alpha == True: pass
            else: #if same_alpha - use same a_r and a_ur for updating of both levers
                self.params_list.append('alpha_r_c')
                if self.RP ==True:
                    self.params_list.append('alpha_ur_c')

            
        if self.forgetting == True and self.reset_to_neutral== False and self.reset_to_mean == False: 
            self.params_list.append('delta')
#             self.model = self.model + '_forgetting'
            
        if self.go_bias == True: 
            self.params_list.append('bias')
            
        if self.repeat == True: 
            self.params_list.append('P')
        
        if self.sensitivity == True: 
            self.params_list.append('rho')
#             self.model = self.model + '_sensitivity'
            
        if self.sticky == True: 
            self.params_list.append('alpha_CK')
            self.params_list.append('beta_CK')
#             self.model = self.model + '_sticky'

    def simulate(self,runs =50, track_var = False):
        sim_df = pd.DataFrame()
        if self.from_fit==True: runs = len(self.fit_df)
        for run in range(runs):
            num_trials = 300
            self.params = []

            if self.from_fit==True: 
                mouse_session = self.fit_df.loc[run,'mouse'] + '_' + str(self.fit_df.loc[run,'session'])
                mouse_model = self.fit_df.loc[run,'mouse'] + '_' + self.model
                params = self.fit_df.loc[run,'params']
                self.params = params
                alpha_r = get_param(self.fit_df,params,'alpha_r')
                beta = get_param(self.fit_df,params,'beta')
                
            else: 
                alpha_r = np.random.beta(1,1); 
                self.params.append(alpha_r)
                beta = np.random.gamma(3,1)
                self.params.append(beta)        
                
                
            if self.RP == True: 
                if self.from_fit==True: alpha_ur = get_param(self.fit_df,params,'alpha_ur')
                else: alpha_ur = np.random.beta(1,1); self.params.append(alpha_ur)
            else: alpha_ur = alpha_r           
            
            
            if self.DSA == True: 
                if self.scale_alpha== True: 
                    if self.from_fit==True: kappa = get_param(self.fit_df,params,'kappa')
                    else: kappa = np.random.beta(1,1); self.params.append(kappa)    
                    alpha_r_c = alpha_r * kappa
                    
                    if self.RP == True:  
                        alpha_ur_c = alpha_ur * kappa
                    else: alpha_ur_c = alpha_r_c

                elif self.same_alpha == True: 
                    alpha_r_c = alpha_r
                    if self.RP == True: alpha_ur_c = alpha_ur
                    else: alpha_ur_c = alpha_r_c

                else:
                    if self.from_fit==True: alpha_r_c = get_param(self.fit_df,params,'alpha_r_c')
                    else: alpha_r_c = np.random.beta(1,1); self.params.append(alpha_r_c)
                    if self.RP == True: 
                        if self.from_fit==True: alpha_ur_c = get_param(self.fit_df,params,'alpha_ur_c')
                        else: alpha_ur_c = np.random.beta(1,1); self.params.append(alpha_ur_c)
                    else: alpha_ur_c = alpha_r_c         
            else: 
                alpha_r_c = 0
                alpha_ur_c = 0

            if self.go_bias == True: 
                if self.from_fit==True: bias = get_param(self.fit_df,params,'bias')
                else: bias = np.random.uniform(low=-1.0, high=1.0); self.params.append(bias)
            else: bias = 0
                
            if self.repeat == True: 
                if self.from_fit==True: P = get_param(self.fit_df,params,'P')
                else: P = np.random.uniform(low=-1.0, high=1.0); self.params.append(P)
            else: P = 0

            if self.sensitivity == True: 
                if self.from_fit==True: rho = get_param(self.fit_df,params,'rho')
                else: rho = np.random.beta(1,1); self.params.append(rho)
            else: rho = 1    
            
            if self.forgetting == True and self.reset_to_neutral== False and self.reset_to_mean == False: 
                if self.from_fit==True: delta = get_param(self.fit_df,params,'delta')
                else: delta = np.random.beta(1,1); self.params.append(delta)
            else: delta =  0

            if self.sticky == True: 
                if self.from_fit==True: 
                    alpha_CK = get_param(self.fit_df,params,'alpha_CK')
                    beta_CK = get_param(self.fit_df,params,'beta_CK')
                    
                else: 
                    alpha_CK = np.random.beta(1,1); self.params.append(alpha_CK)
                    beta_CK = np.random.gamma(3,1); self.params.append(beta_CK)
            else: alpha_CK =  0; beta_CK = 0;
            
            agent = RW_model(alpha_r, alpha_ur, alpha_CK, 
                             beta, beta_CK, bias, rho, 
                            alpha_r_c,alpha_ur_c, delta, P)
            self.params = [alpha_r, alpha_ur, alpha_CK,beta, beta_CK, bias, rho, alpha_r_c,alpha_ur_c, delta,P]
            self.params_list = ['alpha_r', 'alpha_ur', 'alpha_CK','beta', 'beta_CK', 'bias', 'rho', 'alpha_r_c','alpha_ur_c', 'delta','P']


            bandit = Bandit([0.7,0.1])

            switch = int(np.random.choice(range(10,32),1))
       
            actions = np.zeros(num_trials)
            rewards = np.zeros(num_trials)
            optimal = np.zeros(num_trials)
            RPE = np.zeros(num_trials)
            Q_0 = np.zeros(num_trials)
            Q_1 = np.zeros(num_trials)
            Q_ch = np.zeros(num_trials)

            correct_choices = np.zeros(num_trials) 
            track_correct = 0
            
            for t in range(num_trials):
                agent.get_choice_probs()
                
                agent.choose_action() #choose lever
                actions[t] = agent.action #record choice

                bandit.give_reward(agent.action) #give reward
                rewards[t] = bandit.reward #record reward
                Q_0 = agent.Q[0]
                Q_1 = agent.Q[1]
                Q_ch = agent.Q[agent.action]
                RPE[t] = bandit.reward - agent.Q[agent.action]

                agent.update_belief(agent.action,bandit.reward)                    

                bandit.record_optimal_action()
                optimal[t] = bandit.optimal_action*(-2)+1
                correct_choices[t] = (agent.action==bandit.optimal_action)*1
                track_correct += (agent.action==bandit.optimal_action)*1


                if track_correct == switch:
                    bandit.switch()
                    switch = int(np.random.choice(range(10,32),1))
                    track_correct = 0

            if self.from_fit: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials,
                                    'mouse_session': mouse_session, 'mouse_model':mouse_model
                                                       })
            else: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials
                                                       })
                
            if track_var ==True: temp['RPE'] = RPE; temp['Q_0'] = Q_0; temp['Q_1'] = Q_1; temp['Q_ch'] = Q_ch
            sim_df = pd.concat([sim_df,temp])
            sim_df = sim_df.reset_index(drop=True)

        return(sim_df)
    
      
                         
                         

                         
class HMM_sim(object):
    """ simulation of HMM strategy; if set from_fit True, can use parameters from fit_df fits to data,
    can set number of trials in each run and number of simulation runs;
    similar to fits - takes model name and adds augmentations based on the name;
    setting track_var = True keeps track of Qs and RPEs """ 
    def __init__(self, model = 'HMM', fit_df = 'NA',from_fit = False,
                 RP = False, bias = False, sticky = False, fixed_beta = False,
                 fixed_gamma = False,fixed_aur = False,repeat = False):
        
        self.from_fit = from_fit
        if from_fit==True: 
            self.fit_df = fit_df.reset_index(drop = True)
            model = self.fit_df.loc[0,'model']
            
        self.RP = RP; self.bias = bias; self.sticky = sticky; 
        self.fixed_beta = fixed_beta; self.fixed_gamma = fixed_gamma; self.fixed_aur = fixed_aur;self.repeat = repeat;
  
        self.model = model
        
        if 'RP' in model: self.RP = True
        if 'bias' in model: self.bias = True
        if 'sticky' in model: self.sticky = True
        if 'fixed_beta' in model: self.fixed_beta = True
        if 'fixed_gamma' in model: self.fixed_gamma = True
        if 'fixed_aur' in model: self.fixed_aur = True
        if 'repeat' in model: self.repeat = True    
        
    def simulate(self,runs = 10,track_var = False):
        sim_df = pd.DataFrame()
        
        if self.from_fit==True: runs = len(self.fit_df)
            
        for run in range(runs):
            num_trials = 300
            
            if self.from_fit==True: 
                mouse_session = self.fit_df.loc[run,'mouse'] + '_' + str(self.fit_df.loc[run,'session'])
                mouse_model = self.fit_df.loc[run,'mouse'] + '_' + self.model
                params = self.fit_df.loc[run,'params']
                
                alpha_r = get_param(self.fit_df,params,'alpha_r')
            else: 
                alpha_r = np.random.beta(1,1); 


            if self.RP == True: 
                if self.fixed_aur == False:
                    if self.from_fit==True: alpha_ur = get_param(self.fit_df,params,'alpha_ur')
                    else: alpha_ur = np.random.beta(1,1)
                elif self.fixed_aur == True:
                    alpha_ur = 0                
            else: alpha_ur = alpha_r   
                
            
        
            if self.bias == True: 
                if self.from_fit==True: bias = get_param(self.fit_df,params,'bias')
                else: bias = np.random.uniform(low=-1.0, high=1.0)
            else: bias = 0
                
            if self.repeat == True: 
                if self.from_fit==True: P = get_param(self.fit_df,params,'P')
                else: P = np.random.uniform(low=-1.0, high=1.0); 
            else: P = 0

            if self.sticky == True: 
                if self.from_fit==True: 
                    alpha_CK = get_param(self.fit_df,params,'alpha_CK')
                    beta_CK = get_param(self.fit_df,params,'beta_CK')
                    
                else: 
                    alpha_CK = np.random.beta(1,1); 
                    beta_CK = np.random.gamma(3,1); 
            else: alpha_CK =  0; beta_CK = 0;
            
            
            if self.fixed_gamma == True:
                gamma = 0.9; 
            else:
                if self.from_fit==True: 
                    gamma = get_param(self.fit_df,params,'gamma')
                else: 
                    gamma = np.random.beta(1,1); 
            
            if self.fixed_beta == True: 
                beta = 10; 
            else:
                if self.from_fit==True: 
                    beta = get_param(self.fit_df,params,'beta')
                else: 
                    beta = np.random.gamma(3,1); 
     
            agent = HMM_model(alpha_r,alpha_ur,alpha_CK, gamma, beta, beta_CK, bias,P)
            self.params = [alpha_r,alpha_ur,alpha_CK, gamma, beta, beta_CK, bias, P]
            self.params_list = ['alpha_r','alpha_ur','alpha_CK', 'gamma', 'beta', 'beta_CK', 'bias', 'P']
        
            bandit = Bandit([0.7,0.1])

            switch = int(np.random.choice(range(10,32),1))
       
            actions = np.zeros(num_trials)
            rewards = np.zeros(num_trials)
            optimal = np.zeros(num_trials)
            pss_error = np.zeros(num_trials)
            pos_error = np.zeros(num_trials)
        
            belief_error = np.zeros(num_trials)
            pss_0 = np.zeros(num_trials)
            pss_1 = np.zeros(num_trials)
            pos_0 = np.zeros(num_trials)
            pos_1 = np.zeros(num_trials)
            pss_ch = np.zeros(num_trials)
            pos_ch = np.zeros(num_trials)
            
            correct_choices = np.zeros(num_trials) 
            track_correct = 0

            for t in range(num_trials):
                agent.get_choice_probs()
                agent.choose_action()

                actions[t] = agent.action #record choice

                bandit.give_reward(agent.action) #give reward
                rewards[t] = bandit.reward #record reward
                action = agent.action
                reward = bandit.reward
                
                pss_error[t] = bandit.reward - agent.pss[agent.action] 
                
                #pos = prob of observation|state - compare with real outcome to get error 
                pos_error[t] = bandit.reward - agent.pos[agent.action] 

                pss_0[t] = agent.pss[0]
                pss_1[t] = agent.pss[1]
                
                pos_0[t] = agent.pos[0]
                pos_1[t] = agent.pos[1]
                
                pss_ch[t] = agent.pss[agent.action]
                pos_ch[t] = agent.pos[agent.action]

                
                if agent.pss.max() == agent.pss[action]:
                    if reward == 1:
                        belief_error[t] = (1-0.7)

                    if reward == 0:
                        belief_error[t] = (0-0.7)
                else:
                    if reward == 1:
                        belief_error[t] = (1-0.1)
                    if reward == 0:
                        belief_error[t] = (0-0.9)

                agent.update_belief(agent.action,bandit.reward) #updatae CK                          

                bandit.record_optimal_action()
                optimal[t] = bandit.optimal_action*(-2)+1

                correct_choices[t] = (agent.action==bandit.optimal_action)*1
                track_correct += (agent.action==bandit.optimal_action)*1

                if track_correct == switch:
                    bandit.switch()
                    switch = int(np.random.choice(range(10,32),1))
                    track_correct = 0
            
            if self.from_fit: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials,
                                    'mouse_session': mouse_session, 'mouse_model':mouse_model
                                                       })
            else: 
                temp= pd.DataFrame({'model':[self.model]*num_trials,'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                                    'params': [self.params]*num_trials, 'params_list': [self.params_list]*num_trials
                                                       })
                
            if track_var ==True: temp['pss_error']=pss_error; temp['belief_error'] = belief_error; temp['pss_0'] = pss_0; temp['pss_1'] = pss_1; temp['pss_ch'] = pss_ch; temp['pos_error']=pos_error; temp['pos_0'] = pos_0; temp['pos_1'] = pos_1; temp['pos_ch'] = pss_ch
            
            sim_df = pd.concat([sim_df,temp])
            sim_df = sim_df.reset_index(drop=True)

        return(sim_df)
                         

class IO_sim(object):
    def __init__(self, model, beta = 5, alpha_r = 0.7,alpha_ur = 0.2, gamma = 0.9,rew_exp = False, Rm_random = False, RP = False, Q = False, Q_choice = False):
        
        self.rew_exp = rew_exp; self.Rm_random = Rm_random; self.Q = Q; self.Q_choice = Q_choice; self.RP = RP
        
        if 'rew_exp' in model: self.rew_exp = True
        if 'RP' in model: self.RP = True
        if 'Rm_random' in model: self.Rm_random = True
        if 'Q' in model: self.Q = True
        if 'choice' in model: self.Q_choice = True

        self.params_len = 1
        self.params_list = ['beta']
        self.model = 'IO'

        self.beta = beta
        self.gamma = gamma
        
        if self.Q == True: 
            self.params_list.append('alpha_r')
            self.model = self.model + '_Q'
            self.alpha_r = alpha_r
        
            if self.Q_choice ==True: self.model = self.model + '_choice'
            if self.RP ==True: 
                self.model = self.model + '_RP'
                self.alpha_ur = alpha_ur
            else: alpha_ur = alpha_r


            if self.Rm_random ==True: self.model = self.model + '_Rm_random'
        if self.rew_exp ==True: self.model = self.model + '_rew_exp'
        
    def simulate(self,runs):
        sim_df = pd.DataFrame()
        
        for run in range(runs):
            num_trials = 300
     
            agent = IO_model(beta = self.beta, gamma = self.gamma, alpha_r = self.alpha_r,alpha_ur = self.alpha_ur,rew_exp = self.rew_exp, Rm_random = self.Rm_random, Q = self.Q, Q_choice = self.Q_choice)


            bandit = Bandit([0.7,0.1])

            switch = int(np.random.choice(range(10,32),1))
       
            actions = np.zeros(num_trials)
            rewards = np.zeros(num_trials)
            optimal = np.zeros(num_trials)
            
            pss_chosen = np.zeros(num_trials)
            rew_chosen = np.zeros(num_trials)
            pss_0 = np.zeros(num_trials)
            pss_1 = np.zeros(num_trials)
            
            rew_exp_0 = np.zeros(num_trials)
            rew_exp_1 = np.zeros(num_trials)
            
            pss_error = np.zeros(num_trials)
            rew_exp_error = np.zeros(num_trials)

            correct_choices = np.zeros(num_trials) 
            track_correct = 0

            for t in range(num_trials):
                agent.get_choice_probs()
#                 rew_exp_0[t] = agent.rew_exp[0]
#                 rew_exp_1[t] = agent.rew_exp[1]
                
                agent.choose_action()
                pss_chosen[t] = agent.pss[agent.action]
#                 rew_chosen[t] = agent.rew_exp[agent.action]
                
                pss_0[t] = agent.pss[0]
                pss_1[t] = agent.pss[1]
                
                actions[t] = agent.action #record choice
                bandit.give_reward(agent.action) #give reward
                rewards[t] = bandit.reward #record reward
                
                pss_error[t] = bandit.reward - agent.pss[agent.action]
#                 rew_exp_error[t] = bandit.reward - agent.rew_exp[agent.action]
                
                agent.update_belief(agent.action,bandit.reward) #updatae CK                          

                bandit.record_optimal_action()
                optimal[t] = bandit.optimal_action*(-2)+1

                correct_choices[t] = (agent.action==bandit.optimal_action)*1
                track_correct += (agent.action==bandit.optimal_action)*1

                if track_correct == switch:
                    bandit.switch()
                    switch = int(np.random.choice(range(10,32),1))
                    track_correct = 0

            temp= pd.DataFrame({'model':str(self.model)[17:-2],'run':[run]*num_trials,'choice':actions,'reward':rewards,'correct':correct_choices,'transition':optimal,
                               'pss_chosen':pss_chosen,'pss_0':pss_0,'pss_1':pss_1,'pss_error':pss_error,
                                'rew_chosen':rew_chosen,'rew_exp_0':rew_exp_0,'rew_exp_1':rew_exp_1,'rew_exp_error':rew_exp_error})
            sim_df = pd.concat([sim_df, temp])
            sim_df = sim_df.reset_index(drop=True)

        return(sim_df)                
 