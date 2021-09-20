import copy

import numpy as np


from utils import get_coordinate_grid

def generate_particle_distribution(number, major_axis_length, minor_axis_ratios=None, 
                                   major_axis_dir=None, major_plane_normal_dir=None,
                                   margins=None, major_axis_length_stddev=None,
                                   minor_axis_ratios_stddev=None, margins_stddev=None,
                                   random_seed=None):
    """
    Parameters
    ----------
    number : int
        Number of particles to generate.
    major_axis_length : float
        Length of the major axis of the ellipsoid particles. This is taken as the mean
        value if `major_axis_length_stddev` is specified.
    minor_axis_ratios : list of length two of float, optional
        Lengths of the minor axes of the ellipsoid particles, expressed as proportions
        of the `major_axis_length`. By default, `[1, 1]` (i.e. spherical particles). These
        are taken as the mean values if `minor_axis_ratios_stddev` is specified.
    major_axis_dir : list of length three of float, optional
        Direction major axis of the ellipsoid particles. By default, particles are aligned
        along the x-direction: `[1, 0, 0]`. If specified, must be perpendicular to
        `major_plane_normal_dir`.
    major_plane_normal_dir : list of length three of float, optional
        Direction of the major plane normal of the ellipsoid particles. By default, the
        major plane has a normal in the z-direction: `[0, 0, 1]`. If specified, must be
        perpendicular to `major_axis_dir`.
    margins : list of length three of float, optional
        Approximate minimum separation distances allowed between particles. These are
        taken as the mean values if `margins_stddev` is specified. By default, `[0, 0, 0]`
        meaning particles may just touch but not overlap.
    major_axis_length_stddev : float, optional
        Standard deviation of the major axis length. If not specified, all particles will
        have the same major axis length.
    minor_axes_ratio_stddev : list of length two of float, optional
        Standard deviations of the minor axis ratios. If not specified, all particles will
        have the same minor axis ratios.
    margins_stddev : list of length three of float, optional
        Standard deviation of the margins. If not specified, all particles will have the
        same margins.
    random_seed : int, optional
        Seed for random number generation.
        
    Notes
    -----
    Distributions will be truncated to avoid negative values of `major_axis_length`, 
    `minor_axis_ratios` and `margins`.
    
    """
    
    # TODO: avoid negatives in distributions
    
    if minor_axis_ratios is None:
        minor_axis_ratios = [1, 1] # sphere
    if margins is None:
        margins = [0, 0, 0]
    if major_axis_dir is None:
        major_axis_dir = [1, 0, 0]
    if major_plane_normal_dir is None:
        major_plane_normal_dir = [0, 0, 1]            
        
    if major_axis_length_stddev is None:
        major_axis_length_stddev = 0
    if minor_axis_ratios_stddev is None:
        minor_axis_ratios_stddev = [0, 0]
    if margins_stddev is None:
        margins_stddev = [0, 0, 0]
    
    if not np.isclose(np.dot(major_axis_dir, major_plane_normal_dir), 0):
        msg = (f'`major_axis_dir` ({major_axis_dir}) and `major_plane_normal_dir` '
               f'({major_plane_normal_dir}) must be perpendicular.')
        raise ValueError(msg)
    
    rng = np.random.default_rng(random_seed)
    
    major_axis_length_set = rng.normal(
        loc=major_axis_length,
        scale=major_axis_length_stddev,
        size=number,
    )
    minor_axis_ratios_set = np.array([
        rng.normal(loc=minor_axis_ratios[0], scale=minor_axis_ratios_stddev[0], size=number),
        rng.normal(loc=minor_axis_ratios[1], scale=minor_axis_ratios_stddev[1], size=number),       
    ]).T        
    margins_set = np.array([
        rng.normal(loc=margins[0], scale=margins_stddev[0], size=number),
        rng.normal(loc=margins[1], scale=margins_stddev[1], size=number),
        rng.normal(loc=margins[2], scale=margins_stddev[2], size=number),
    ]).T
    
    particles = [
        {
            'major_axis_length': major,
            'minor_axis_ratios': minor,
            'margins': marg,
            'major_axis_dir': major_axis_dir,
            'major_plane_normal_dir': major_plane_normal_dir,
        }
        for major, minor, marg in zip(
            major_axis_length_set,
            minor_axis_ratios_set,
            margins_set,
        )
    ]
    
    return particles

class Particle:

    def __init__(self, major_axis_length, centre, minor_axis_ratios=None, margins=None,
                major_axis_dir=None, major_plane_normal_dir=None):
        """
        Parameters
        ----------
        major_axis_length : number                
        centre : list of number
        minor_axis_ratios : list of number, optional
        
        """
        
        if major_axis_dir is None:
            major_axis_dir = [1, 0, 0]
        if major_plane_normal_dir is None:
            major_plane_normal_dir = [0, 0, 1]
        if minor_axis_ratios is None:
            minor_axis_ratios = [1, 1]
        if margins is None:
            margins = [0, 0, 0]
                
        self.major_axis_length = major_axis_length
        self._centre_history = []
        self.centre = centre
        self.minor_axis_ratios = minor_axis_ratios
        self.margins = margins
        self.major_axis_dir = major_axis_dir
        self.major_plane_normal_dir = major_plane_normal_dir
        
        self._validate()
        
        
    def _validate(self):
        for ratio in self.minor_axis_ratios:
            if ratio > 1:
                msg = (f'Minor axes ratios should be less than 1, but values are: '
                       f'{self.minor_axis_ratios}')
                raise ValueError(msg)
        
    @property
    def centre(self):
        return self._centre
    
    @centre.setter
    def centre(self, new_centre):
        self._centre = new_centre
        self._centre_history.append(self._centre)               
        
    @property
    def centre_history(self):
        return self._centre_history
    
    @property
    def minor_axes_lengths(self):
        return (
            self.major_axis_length * self.minor_axis_ratios[0],
            self.major_axis_length * self.minor_axis_ratios[1],
        )
    
    @property
    def axes_sizes(self):
        return tuple([self.major_axis_length] + list(self.minor_axes_lengths))
    
    @classmethod
    def from_axis_sizes(cls, axis_sizes, centre, margins=None):
        maj_ax_len = axis_sizes[0]
        min_ax_ratios = [
            axis_sizes[1] / maj_ax_len,
            axis_sizes[2] / maj_ax_len,
        ]
        particle = cls(
            major_axis_length=maj_ax_len,
            centre=centre,
            minor_axis_ratios=min_ax_ratios,
            margins=margins
        )
        return particle
    
    @property
    def margined_particle(self):
        """Get the expanded particle, including margins."""        
        new_axes_sizes = [i + j for i, j in zip(self.axes_sizes, self.margins)]
        particle = Particle.from_axis_sizes(new_axes_sizes, centre=self.centre)
        return particle

    def displace(self, displacement):
        self.centre += displacement
    
class ParticleRVE:
    
    def __init__(self, size, grid_size, particles=None, random_seed=None):
        """
        Parameters
        ----------
        size
        grid_size
        particles : list of (dict or Particle)
            Dict with the following keys:
                major_axis_size : number                
                centre : list of number
                minor_axes_ratio : list of number, optional                                
        """
        self.size = size
        self.grid_size = grid_size                                    
        self.grid_obj = self._generate_damask_grid_obj()
        self.particles = []
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed) # TODO need to make this seed different to particle seed?        
        
        self._add_particles(particles)
                    
    def _add_particles(self, particles):
        new_particles = []
        for particle in particles or []:
            if not isinstance(particle, Particle):
                particle = Particle(**particle)
            
            if particle.major_axis_length > self.size[0]: # todo: for now assume x-axis is major axis
                msg = (f'Major axis size of particle ({particle.major_axis_length}) must '
                f'be smaller than major axis size of RVE {self.size[0]}.')
                raise ValueError(msg)
            
            new_particles.append(particle)
        
            self._add_particle(particle)
    
    @property
    def material(self):
        return self.grid_obj.material
    
    @classmethod
    def from_particle_distribution(cls, size, grid_size, number, major_axis_length,
                                   minor_axis_ratios=None, major_axis_dir=None,
                                   major_plane_normal_dir=None, margins=None,
                                   major_axis_length_stddev=None,
                                   minor_axis_ratios_stddev=None, margins_stddev=None,
                                   random_seed=None):
        
        particles = generate_particle_distribution(
            number=number,
            major_axis_length=major_axis_length,
            minor_axis_ratios=minor_axis_ratios,
            major_axis_dir=major_axis_dir,
            major_plane_normal_dir=major_plane_normal_dir,
            margins=margins,
            major_axis_length_stddev=major_axis_length_stddev,
            minor_axis_ratios_stddev=minor_axis_ratios_stddev,
            margins_stddev=margins_stddev,
            random_seed=random_seed,
        )
        part_RVE = cls(size=size, grid_size=grid_size, random_seed=random_seed)
        part_RVE.add_particles(particles)
        
        return part_RVE
    
    def _get_random_centre(self):                
        centre = self.rng.random(3,)
        centre[0] *= self.size[0]
        centre[1] *= self.size[1]
        centre[2] *= self.size[2]
        return centre
    
    def _generate_damask_grid_obj(self):
        from damask import seeds, Grid
        my_seeds = seeds.from_random([1, 1, 1], 1) # todo better way to do this?
        grid_obj = Grid.from_Voronoi_tessellation(
            cells=np.array(self.grid_size),    
            size=self.size,
            seeds=my_seeds,    
            periodic=True,
        )
        return grid_obj
        
    def _add_particle(self, particle):        
        grid_obj = self.grid_obj.add_primitive(
            dimension=particle.axes_sizes,
            center=particle.centre,
            exponent=1,
        )
        self.particles.append(particle)
        self.grid_obj = grid_obj        
        
    @property
    def num_voxels(self):
        return np.product(self.grid_size)
        
    def add_particle(self, particle):
        """Add a particle to the RVE."""
                
        if isinstance(particle, Particle):
            # Add without modification:
            self._add_particle(particle)
            
        else:
            # Iterate on position to maintain margins between particles:
            particle = Particle(**particle, centre=self._get_random_centre())            
            single_particle_RVE = ParticleRVE(
                self.size,
                self.grid_size,
                particles=[particle.margined_particle],
            )
            
            overlap_bool = np.logical_and(
                self.material != 0,
                single_particle_RVE.material != 0
            )
            single_particle_RVE_voxels = single_particle_RVE.material != 0                
            overlap_fraction = np.sum(overlap_bool) / np.sum(single_particle_RVE_voxels)            
            count = 0
            while np.any(overlap_bool):
                if count > 200:
                    print('Cannot find suitable position for particle.')
                    return
                
                # Try a new position:
                particle.centre = self._get_random_centre()
                single_particle_RVE = ParticleRVE(
                    self.size,
                    self.grid_size,
                    particles=[particle.margined_particle],
                )
                overlap_bool = np.logical_and(
                    self.material != 0,
                    single_particle_RVE.material != 0
                )    
                single_particle_RVE_voxels = single_particle_RVE.material != 0                
                overlap_fraction = np.sum(overlap_bool) / np.sum(single_particle_RVE_voxels)
                count += 1
            
            if count > 0:
                print(f'Found new particle centre in {count} iterations.')
                
            self._add_particle(particle)
    
    def add_particles(self, particles):
        for particle in particles:
            self.add_particle(particle)    

    def save(self, *args, **kwargs):
        """Call save on the DAMASK Grid object."""
        self.grid_obj.save(*args, **kwargs)
            
    def write_VTR_particle_history(self, directory):
        """Generate a series of VTR files showing the insertion of the particles."""
        
        base_grid = self._generate_damask_grid_obj()        
        num_inserts = sum([len(particle.centre_history) for particle in self.particles])
        zero_pad_len = len(str(num_inserts))
        count = 0
        base_grid.save(f'{directory}/particle_RVE_{count:0{zero_pad_len}}.vtr')        
        for particle in self.particles:              
            for historic_centre in particle.centre_history:
                count += 1
                grid = base_grid.add_primitive(
                    dimension=particle.axes_sizes,
                    center=historic_centre,
                    exponent=1,
                )
                grid.save(f'{directory}/particle_RVE_{count:0{zero_pad_len}}.vtr')        
            base_grid = grid
    
    def show_slice(self):
        from plotly import graph_objects
        data = [
            {
                'type': 'heatmap',
                'z': self.grid_obj.material[int(self.grid_size[0]/2)],
            }
        ]
        layout = {
            'xaxis': {
                'scaleanchor': 'y',
                'constrain': 'domain',
            }
        }
        fig = graph_objects.FigureWidget(data=data, layout=layout)        
        return fig

    def show(self):        
        from plotly import graph_objects
        # TODO: fix
        coords = get_coordinate_grid(self.size, self.grid_size)
        coords_flat = coords[0].reshape(-1, 3)
        material_flat = self.grid_obj.material.reshape(-1)
        matrix_idx = material_flat == 0
        value = material_flat[~matrix_idx]
        coords_flat = coords_flat[~matrix_idx]
        data = [
            {
                'type': 'volume',
                'x': coords_flat[:, 0],
                'y': coords_flat[:, 1],
                'z': coords_flat[:, 2],
                'value': value,
                'surface_count': len(self.particles),
                'opacity': 0.1,
            },
        ]
        layout = {
            'xaxis': {},
        }
        fig = graph_objects.FigureWidget(data=data, layout=layout)        
        return fig        
    