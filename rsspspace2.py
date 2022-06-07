import numpy as np
from scipy.stats import qmc
from scipy.stats import special_ortho_group
from scipy.optimize import minimize

import warnings

class RSSPSpace2:
    def __init__(self,  domain_dim:int,ssp_dim: int, axis_matrix=None, phase_matrix=None,
                 domain_bounds=None, length_scale=lambda x: np.ones(x.shape)): 
        self.domain_dim = domain_dim
        self.ssp_dim = ssp_dim
        self.length_scale = length_scale
        if domain_bounds is not None:
            assert domain_bounds.shape[0] == domain_dim
        self.domain_bounds = domain_bounds
        if (axis_matrix is None) & (phase_matrix is None):
            raise RuntimeError("RSSP spaces must be defined by either an axis matrix or phase matrix function.")
        elif (phase_matrix is None):
            if callable(axis_matrix):
                self.phase_matrix = lambda x: (-1.j*np.log(np.fft.fft(axis_matrix(x),axis=0))).real
                self.axis_matrix = axis_matrix
            else:
                assert axis_matrix.shape[0] == ssp_dim
                assert axis_matrix.shape[1] == domain_dim
                self.phase_matrix = lambda x: (-1.j*np.log(np.fft.fft(axis_matrix,axis=0))).real
                self.axis_matrix = lambda x: axis_matrix
        elif (axis_matrix is None):
            if callable(phase_matrix):
                self.axis_matrix = lambda x: np.fft.ifft(np.exp(1.j*phase_matrix(x)), axis=0).real
                self.phase_matrix = phase_matrix
            else:
                assert phase_matrix.shape[0] == ssp_dim
                assert phase_matrix.shape[1] == domain_dim
                self.axis_matrix =lambda x:np.fft.ifft(np.exp(1.j*phase_matrix), axis=0).real
                self.phase_matrix = lambda x: phase_matrix
                 
    def update_lengthscale(self, scale):
        self.length_scale  = scale
        
    def encode(self,x):
        assert x.ndim == 2, f'Expected 2d data (samples, features), got {x.ndim}d data.'
        assert x.shape[1] == self.domain_dim
        #data = np.fft.ifft( np.exp( 1.j * self.phase_matrix(x) @ (x / self.length_scale(x)).T), axis=0 ).real
        
        data = np.zeros((self.ssp_dim,x.shape[0]))
        scaled_x = self.length_scale(x)
        for i in range(x.shape[0]):
            #print(x[i,:].shape)
            data[:,i] = np.fft.ifft(np.prod(np.fft.fft(self.axis_matrix(x[i,:]), axis=0)**scaled_x[i,:], axis=1), axis=0).real
        return data.T
   
    
    def encode_fourier(self,x):
        assert x.ndim == 2, f'Expected 2d data (samples, features), got {x.ndim} data.'
        data =  np.exp( 1.j * self.phase_matrix(x) @ (x / self.length_scale(x)).T )
        return data.T
                 
    def get_sample_points(self,num_points,method='grid'):
        if self.domain_bounds is None:
            bounds = np.vstack([-10*np.ones(self.domain_dim), 10*np.ones(self.domain_dim)]).T
        else:
            bounds = self.domain_bounds
        if method=='grid':
            n_per_dim = int(num_points**(1/self.domain_dim))
            if n_per_dim**self.domain_dim != num_points:
                warnings.warn((f'Evenly distributing points over a '
                               f'{self.domain_dim} grid requires numbers '
                               f'of samples to be powers of {self.domain_dim}.'
                               f'Requested {num_points} samples, returning '
                               f'{n_per_dim**self.domain_dim}'), RuntimeWarning)
            ### end if
            xs = np.linspace(bounds[:,0],bounds[:,1],n_per_dim)
            xxs = np.meshgrid(*[xs[:,i] for i in range(self.domain_dim)])
            retval = np.array([x.reshape(-1) for x in xxs]).T
            assert retval.shape[1] == self.domain_dim, f'Expected {self.domain_dim}d data, got {retval.shape[1]}d data'
            return retval
        elif method=='sobol':
            sampler = qmc.Sobol(d=self.domain_dim) 
            lbounds = bounds[:,0]
            ubounds = bounds[:,1]
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
        else:
            raise NotImplementedError()
        return sample_points.T 
        
    
    def get_sample_ssps(self,num_points,method='grid'): 
        sample_points = self.get_sample_points(num_points,method=method)
        sample_ssps = self.encode(sample_points)
        return sample_ssps
    
    def get_sample_pts_and_ssps(self,num_points,method='grid'): 
        sample_points = self.get_sample_points(num_points,method)
        expected_points = (int(num_points**(1/self.domain_dim))**self.domain_dim)
        assert sample_points.shape[0] == expected_points, f'Expected {expected_points} samples, got {sample_points.shape[0]}.'

        sample_ssps = self.encode(sample_points)
        assert sample_ssps.shape[0] == expected_points

        return sample_ssps, sample_points
    
    def normalize(self,ssp):
        return ssp/np.sqrt(np.sum(ssp**2))
    
    def make_unitary(self,ssp):
        fssp = np.fft.fft(ssp)
        fssp = fssp/np.sqrt(fssp.real**2 + fssp.imag**2)
        return np.fft.ifft(fssp).real  
    
    def identity(self):
        s = np.zeros(self.ssp_dim)
        s[0] = 1
        return s
    
    def bind(self,a,b):
        return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real
    
    def invert(self,a):
        return a[-np.arange(len(a))]
                 
                 
    def decode(self,ssp,method='from-set',sampling_method='grid',
               num_samples =1000, samples=None): # other args for specfic methods
        if samples is None:
            sample_ssps, sample_points = self.get_sample_pts_and_ssps(num_samples,sampling_method)
        else:
            sample_ssps, sample_points = samples
            
        assert sample_ssps.shape[1] == ssp.shape[1]
        
        sims = sample_ssps @ ssp.T
        return sample_points[np.argmax(sims),:]
        
        
    def clean_up(self,ssp,method='from-set',num_samples =1000,sampling_method='grid'):
        sample_ssps = self.get_sample_ssps(num_samples,method=method)
        sims = sample_ssps.T @ ssp
        return sample_ssps[:,np.argmax(sims)]
    
    def similarity_plot(self,ssp,n_grid=100,plot_type='heatmap',ax=None,**kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
        if self.domain_dim == 1:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            sims = ssp @ self.encode(np.atleast_2d(xs).T).T
            im = ax.plot(xs, sims.reshape(-1) )
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
        elif self.domain_dim == 2:
            xs = np.linspace(self.domain_bounds[0,0],self.domain_bounds[0,1], n_grid)
            ys = np.linspace(self.domain_bounds[1,0],self.domain_bounds[1,1], n_grid)
            X,Y = np.meshgrid(xs,ys)
            sims = ssp @ self.encode(np.vstack([X.reshape(-1),Y.reshape(-1)]).T).T 
            if plot_type=='heatmap':
                im = ax.pcolormesh(X,Y,sims.reshape(X.shape),**kwargs)
            elif plot_type=='contour':
                im = ax.contour(X,Y,sims.reshape(X.shape),**kwargs)
            elif plot_type=='contourf':
                im = ax.contourf(X,Y,sims.reshape(X.shape),**kwargs)
            ax.set_xlim(self.domain_bounds[0,0],self.domain_bounds[0,1])
            ax.set_ylim(self.domain_bounds[1,0],self.domain_bounds[1,1])
        else:
            raise NotImplementedError()
        return im
