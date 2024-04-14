"""
modified from https://github.com/alisiahkoohi/Langevin-dynamics
"""

import torch

from .SGLD import SGLD
import copy
import numpy as np

class LangevinDynamics(object):
    """
    LangevinDynamics class for performing Langevin dynamics optimization.

    Args:
        x (torch.Tensor): Initial parameter values.
        func (callable): The loss function to be optimized.
        lr (float, optional): Initial learning rate. Default is 1e-2.
        lr_final (float, optional): Final learning rate. Default is 1e-4.
        max_itr (int, optional): Maximum number of iterations. Default is 1e4.
        device (str, optional): Device to perform computations on ('cpu' or
            'cuda'). Default is 'cpu'.
        base_dist: (object, optional) used for generating GP noise
        temperature: (int, optional) SGLD temperature
        momentum : (int, optional), momentum in SGD
        use_GP_noise: (bool, optional), use base_dist to generate GP noise when True

    """

    def __init__(self,
                 x: torch.Tensor,
                 func: callable,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: int = 10000,
                 device: str = 'cpu',
                 base_dist=None,
                 temperature=1,
                 momentum=0,
                 use_GP_noise=False):
        super(LangevinDynamics, self).__init__()
        print('update')
        self.x = x
        #self.x_noise = None
        print("Tempertaure:{}".format(temperature))
        self.optim = SGLD([self.x], lr, weight_decay=0.0, device=device, temperature=temperature, momentum=momentum)
            
        if (base_dist is not None) & (use_GP_noise==True):
            samples_all = []
            with torch.no_grad():
                for i in range(int(np.ceil(max_itr/10000))):
                    samples_all.append(base_dist.rsample(sample_shape=(10000,))[:,:,None].cpu())
            self.x_noise = torch.vstack(samples_all)
            self.base_dist = base_dist
        
        if use_GP_noise == False:
            print('start, white noise on A')
        else: 
            print('start, GP noise on A')
        self.lr = lr
        self.lr_final = lr_final
        self.max_itr = max_itr
        self.func = func
        self.lr_fn = self.decay_fn(lr=lr, lr_final=lr_final, max_itr=max_itr)
        self.counter = 0.0
        self.device = device
        self.use_GP_noise = use_GP_noise

    # optional for GP noise injection
    def sample_a_noise(self):
        return self.base_dist.rsample(sample_shape=(1,))[:,:,None]
        
    def sample(self, epoch) -> tuple:
        """
        Perform a Langevin dynamics step.

        Returns:
            tuple: A tuple containing the current parameter values and the loss
                value.
        """
        self.lr_decay()
        self.optim.zero_grad()
        loss = self.func(self.x)
        loss.backward()
        
        if self.use_GP_noise == False:
            self.optim.step()
        else:
            self.optim.step(input_noise=self.x_noise[epoch:epoch+1].to(self.device))
        #self.optim.step(input_noise=self.sample_a_noise())
        self.counter += 1
        return copy.deepcopy(self.x.data), loss.item()

    def decay_fn(self,
                 lr: float = 1e-2,
                 lr_final: float = 1e-4,
                 max_itr: int = 10000) -> callable:
        """
        Calculate the learning rate decay function.

        Args:
            lr (float): Initial learning rate.
            lr_final (float): Final learning rate.
            max_itr (int): Maximum number of iterations.

        Returns:
            callable: Learning rate decay function.
        """
        gamma = -0.55
        b = max_itr / ((lr_final / lr)**(1 / gamma) - 1.0)
        a = lr / (b**gamma)

        def lr_fn(t: float,
                  a: float = a,
                  b: float = b,
                  gamma: float = gamma) -> float:
            """
            Calculate the learning rate based on the iteration number.

            Args:
                t (float): Current iteration number.
                a (float): Scaling factor.
                b (float): Scaling factor.
                gamma (float): Exponent factor.

            Returns:
                float: Learning rate at the given iteration.
            """
            #print('t :{}, a :{}, b:{}, gamma:{}'.format(t,a,b,gamma))
            return a * ((b + t)**gamma)
        
        #print('lr_fn(0) :{}'.format(lr_fn(0)))
        return lr_fn

    def lr_decay(self):
        """
        Update the learning rate of the optimizer based on the current
        iteration.
        """
        for param_group in self.optim.param_groups:        
            param_group['lr'] = self.lr_fn(self.counter)
            #print('after lr : {}'.format(param_group['lr']))

