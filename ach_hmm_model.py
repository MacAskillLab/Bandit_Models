import numpy as np
import pandas as pd


class Bandit:
    def __init__(self, probs):
        self.probs = list(probs)          # e.g., [0.7, 0.3]
        self.n = 2
        self.outcomes = [0, 1]

    def give_reward(self, action):
        self.reward = int(np.random.choice(self.outcomes, 1,
                                           p=[1 - self.probs[action], self.probs[action]])[0])

    def record_optimal_action(self):
        self.optimal_action = int(np.argmax(self.probs))

    def switch(self):
        self.probs = [self.probs[1], self.probs[0]]


class HMM_model:
    def __init__(self, alpha_r, alpha_ur, gamma, beta=10.0,
                 alpha_CK=0.0, beta_CK=0.0, bias=0.0, P=0.0,
                 p_sens=1.0, l_sens=1.0, p_mode='linear', l_mode='linear'):
        # core params
        self.alpha_r  = float(alpha_r)
        self.alpha_ur = float(alpha_ur)
        self.gamma    = float(gamma)
        self.beta     = float(beta)
        # extras
        self.alpha_CK = float(alpha_CK)
        self.beta_CK  = float(beta_CK)
        self.bias     = float(bias)
        self.P        = float(P)
        # sensitivities
        self.p_sens   = float(p_sens)     # prior sensitivity (1=use prior as-is; <1 => more certainty)
        self.l_sens   = float(l_sens)     # likelihood sensitivity (1=as-is)
        self.p_mode   = p_mode
        self.l_mode   = l_mode

        # state
        self.n = 2
        self.pss = np.array([0.5, 0.5])   # prior p(state)
        self.pos = np.array([0.5, 0.5])   # likelihood p(o|state) for current trial
        self.CK  = np.zeros(2)
        self.action = 10                  # sentinel for first trial

        # transition matrix (symmetric, strength set by gamma)
        t0 = t1 = ts = 0.5
        self.Ts = np.array([[t0 + ts*self.gamma, t1 - ts*self.gamma],
                            [t1 - ts*self.gamma, t0 + ts*self.gamma]])

    # ----- helpers -----
    def _apply_p_sens(self, p):
        """certainty-boosting: p_sens=1 => identity; p_sens<1 => pushes toward the mode (more certain)."""
        if self.p_sens == 1.0: return p
        best = int(np.argmax(p))
        s = 1.0 - self.p_sens*(1.0 - p[best])
        return np.array([s, 1.0 - s]) if best == 0 else np.array([1.0 - s, s])

    def _map_alpha(self, a):  
        """Apply likelihood sensitivity scaling to alpha parameters."""
        if self.l_mode == 'linear':
            return a * self.l_sens
        elif self.l_mode == 'log':
            # For 'log', we'll apply scaling to the likelihood distribution itself in update_belief
            return a
        else:
            raise ValueError(f"Unknown l_mode: {self.l_mode}")

    # ----- policy / update -----
    def get_choice_probs(self):
        V = self.pss.copy()
        V[0] += self.bias
        if self.action != 10:             # repetition bias if you ever set P != 0
            V[self.action] += self.P
        logits = V*self.beta + self.CK*self.beta_CK
        logits -= logits.max()
        e = np.exp(logits)
        self.probs = e / e.sum()

    def choose_action(self):
        self.action = int(np.random.choice([0,1], p=self.probs))

    def update_belief(self, action, reward):
        a_r  = self._map_alpha(self.alpha_r)
        a_ur = self._map_alpha(self.alpha_ur)
        t0 = t1 = ts = 0.5

        if reward == 1:
            self.pos[action]     = t0 + ts*a_r
            self.pos[1 - action] = t1 - ts*a_r
        else:
            self.pos[action]     = t1 - ts*a_ur
            self.pos[1 - action] = t0 + ts*a_ur

        s_pss = self._apply_p_sens(self.pss)    # prior sensitivity
        ps = s_pss * self.pos                   # Bayes (unnormalized)
        ps /= ps.sum()
        self.pss = ps @ self.Ts                 # one-step state transition

        # optional recency CK (off by default)
        if self.alpha_CK > 0:
            self.CK *= (1.0 - self.alpha_CK)
            self.CK[action] += self.alpha_CK


def simulate_session(runs=20, num_trials=500, probs=(0.7,0.3),
                     alpha_r=0.62, alpha_ur=0.10, gamma=0.45,
                     beta=10.0, sticky=False, repeat=False,
                     p_sens=1.0, l_sens=1.0, seed=11):

    rng = np.random.default_rng(seed)
    rows = []

    for run in range(runs):
        alpha_CK = rng.beta(2,5) if sticky else 0.0
        beta_CK  = rng.gamma(2,1) if sticky else 0.0
        P        = rng.uniform(-0.2,0.2) if repeat else 0.0
        bias     = 0.0

        agent  = HMM_model(alpha_r, alpha_ur, gamma, beta,
                           alpha_CK, beta_CK, bias, P,
                           p_sens=p_sens, l_sens=l_sens)

        bandit = Bandit(probs)
        track_correct = 0
        switch_after = rng.integers(10, 32)     # performance-driven like your mice

        agent.get_choice_probs()
        
        for t in range(num_trials):
            agent.get_choice_probs()
            agent.choose_action()
            bandit.give_reward(agent.action)
            bandit.record_optimal_action()

            rows.append({
                "run": run, "trial": t,
                "choice": int(agent.action),
                "reward": int(bandit.reward),
                "optimal": int(bandit.optimal_action),
                "post_0": float(agent.pss[0]),
                "post_1": float(agent.pss[1]),
                "p_sens": p_sens, "sticky": sticky
            })

            correct = int(agent.action == bandit.optimal_action)
            track_correct += correct
            agent.update_belief(agent.action, bandit.reward)

            if track_correct == switch_after:
                bandit.switch()
                switch_after = rng.integers(10, 32)
                track_correct = 0

    df = pd.DataFrame(rows)
    df["task"] = f"{int(probs[0]*100)}:{int(probs[1]*100)}"
    return df


