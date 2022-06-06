from audioop import mul
from turtle import forward
from typing import Union

import numpy as np

from .prior import Prior
from .pdf import PDF

T = Union[Prior, np.ndarray]
INF = 1e100

class Model(PDF):
    """
    This class represents a predictive model p(x|w), such as N(x|mu,sigma).

    You can use this class as not only point estimation statistics but also bayesian statistics.

    """
    def __init__(self):
        super().__init__()

class _LDR(Model):
    def __init__(self, w, x, mu, sigma):
        """
        This class represents a Linear Dimensionaly Reduction model.
        Args:
            
            W ((M,D) shape T): 
            x ((N,M) shape T): 
            mu ((D) shape T): 
            sigma (float): 
        """
        super().__init__()
        self.w = w
        self.x = x
        self.mu = mu
        self.sigma = sigma

class LDR(_LDR):
    """
    You can use this class as bayesian statistics.
    """
    def __init__(self, y, valid, w, x, mu, sigma, w_mean, w_variance, x_mean, x_variance, mu_mean, mu_variance):
        """
        Additional parameters, *_mean and *_variance, are used in optimization part

        Args:
            y ((N,D) shape np.ndarray): 
            valid ((M,D) shape np.ndarray): 
            w_mean ((M,D) shape np.ndarray): 
            w_variance ((M,M) shape np.ndarray): w[:,d] ~ N(w_mean[:,d], w_variance)
            x_mean ((N,M) shape np.ndarray): 
            x_variance ((M,M) shape np.ndarray): x[n,:] ~ N(x_mean[n,:], x_variance)
            mu_mean ((D) shape np.ndarray): 
            mu_variance ((D,D) shape np.ndarray): mu ~ N(mu_mean, mu_variance)
        """
        super().__init__(w, x, mu, sigma)
        self.y = y
        self.valid = valid
        self.w_mean = w_mean
        self.w_variance = w_variance
        self.x_mean = x_mean
        self.x_variance = x_variance
        self.mu_mean = mu_mean
        self.mu_variance = mu_variance

        self.n = self.y.shape[0] # sample size
        self.d = self.y.shape[1] # full dimension
        self.m = self.w.mean.shape[0] # partial dimension

        self.priori_inv_mu_var = np.linalg.inv(self.mu.variance)
        self.priori_inv_w_var = np.linalg.inv(self.w.variance)

        self.tr_best = None
        self.va_best = None
        self.loss_tr = INF
        self.loss_va = INF
        self.log_tr = []
        self.log_va = []

    def predict(self, y, is_best=False, is_tr = False):
        if not is_best:
            w = self.w_mean
            x_var = self.x_variance
            mu = self.mu_mean
        else:
            if is_tr:
                dic = self.tr_best
            else:
                dic = self.va_best

            w = dic["w_mean"]
            x_var = dic["x_variance"]
            mu = dic["mu_mean"]
        x = (self.sigma**(-2) * x_var @ w @ (y - mu).T).T
        
        wtx = (w.T@x.T).T
        return wtx + mu

    def calculate_logloss(self, y, is_best=False, is_tr = False):
        mean = y - (self.predict(y=y))
        mat_res = np.sum((mean@ np.linalg.inv(self.sigma**(-2)*np.eye(self.d))) * mean,axis = 1)
        return np.mean(mat_res)

    def forward(self, y=None, is_train = True):
        """
        Args:
            y (N, D):
                This is input
            is_train (bool, optional):
                If the value is true, it uses x_mean for calculation directly.
                If the value is false, it doesn't use x_mean for calculation directly.
                
                Defaults to True.
        """
        return None

    def optimize(self, epoch, valid = None, order = 0, is_print = True):
        """
        This order is following (6 pattern):
            0: mu -> w -> x

        Args:
            epoch (int):
            valid ((NN,D) shape np.ndarray):
        """
        if order == 0:
            for i in range(epoch):
                self.calc_mu()
                self.calc_w()
                self.calc_x()

                self.memorize(i, valid, is_print)
        return None

    def calc_mu(self):
        self.mu_variance = np.linalg.inv(
            self.n * (self.sigma**(-2)) * np.eye(self.d) +self.priori_inv_mu_var
        )
        self.mu_mean = self.sigma**(-2) * self.mu_variance \
            @ np.sum(self.y - (self.w_mean.T@ self.x_mean.T).T , axis = 0)

    def calc_w(self):
        _xxt = np.zeros((self.m, self.m))
        for i in range(self.n):
            _xxt += self.x_mean[i].reshape(-1,1)@self.x_mean[i].reshape(-1,1).T
        self.w_variance = np.linalg.inv(
            self.sigma**(-2) * (_xxt +self.x_variance) + self.priori_inv_w_var
        )
        _m_w = np.empty((self.m, self.d))
        _diff = self.y - self.mu_mean
        for d in range(self.d):
            _m_w[:,d] = self.sigma**(-2) \
                * self.w_variance \
                    @ np.sum(_diff[:,d] \
                        * self.x_mean.T, axis = 1)
        self.w_mean = _m_w
    
    def calc_x(self):
        _wwt = 0
        for d in range(self.d):
            _wwt += self.w_mean[:,d].reshape(-1,1) @ self.w_mean[:,d].reshape(-1,1).T
        self.x_variance = np.linalg.inv(
            self.sigma**(-2) \
                * (_wwt + self.w_variance) \
                        + np.eye(self.m)
        )
        self.x_mean = (self.sigma**(-2) * self.x_variance @ self.w_mean @ (self.y - self.mu_mean).T).T

    def memorize(self, i, valid, is_print):
        """
        Args:
            i (_type_): _description_
            valid (_type_): _description_
            is_print (bool): _description_
        """
        tr_loss = self.calculate_logloss(y=self.y)
        va_loss = self.calculate_logloss(y=self.valid)

        if tr_loss < self.loss_tr:
            self.tr_best = self.save(True)
            self.loss_tr = tr_loss
        if va_loss < self.loss_va:
            self.va_best = self.save()
            self.loss_va = va_loss
        
        self.log_tr.append(tr_loss)
        self.log_va.append(va_loss)
        
        if is_print:
            print(f"Epoch{i}: train{tr_loss:.4g}, valid{va_loss:.4g}")
    
    def save(self, is_tr = False):
        _t = {
            "w_mean": self.w_mean,
            "w_variance": self.w_variance,
            "x_mean": self.x_mean,
            "x_variance": self.x_variance,
            "mu_mean": self.mu_mean,
            "mu_variance": self.mu_variance,
        }
        if is_tr:
            self.tr_best = _t
        else:
            self.va_best = _t
        return None
