"""
TODO: 
    - Rethink to/from_JSON_like
    - Ensure we can reproduce the same RVE with random_seed specification
"""

import copy

import numpy as np

from damask_parse.utils import perpendicular_vectors


from damask_parse.utils import get_coordinate_grid


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
        minor_axis_ratios = [1, 1]  # sphere
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
        rng.normal(loc=minor_axis_ratios[0],
                   scale=minor_axis_ratios_stddev[0], size=number),
        rng.normal(loc=minor_axis_ratios[1],
                   scale=minor_axis_ratios_stddev[1], size=number),
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

    def __init__(self, major_axis_length, centre=None, minor_axis_ratios=None, margins=None,
                 major_axis_dir=None, major_plane_normal_dir=None, distribution=None,
                 _centre_history=None):
        """
        Parameters
        ----------
        major_axis_length : number                
        centre : list of number
        minor_axis_ratios : list of number, optional
        distribution : ParticleDistribution, optional
            ParticleDistribution from which this particle was generated.
        """

        if major_axis_dir is None:
            major_axis_dir = [1, 0, 0]
        if major_plane_normal_dir is None:
            major_plane_normal_dir = [0, 0, 1]
        if minor_axis_ratios is None:
            minor_axis_ratios = [1, 1]
        if margins is None:
            margins = [0, 0, 0]
        if centre is None:
            centre = [0, 0, 0]

        self.major_axis_length = major_axis_length
        self._centre_history = _centre_history or []
        self.centre = centre
        self.minor_axis_ratios = minor_axis_ratios
        self.margins = margins
        self.major_axis_dir = major_axis_dir
        self.major_plane_normal_dir = major_plane_normal_dir
        self.distribution = distribution

        self._validate()

    def to_JSON_like(self):
        dct = {
            'major_axis_length': self.major_axis_length,
            'minor_axis_ratios': self.minor_axis_ratios,
            'centre': self.centre,
            'margins': self.margins,
            'major_axis_dir': self.major_axis_dir,
            'major_plane_normal_dir': self.major_plane_normal_dir,
            '_centre_history': [list(i) for i in self._centre_history],
        }
        return dct

    @classmethod
    def from_JSON_like(cls, dct, distribution=None):
        dct = copy.deepcopy(dct)
        return cls(**dct, distribution=distribution)

    def _validate(self):
        for ratio in self.minor_axis_ratios:
            if ratio > 1:
                msg = (f'Minor axes ratios should be less than 1, but values are: '
                       f'{self.minor_axis_ratios}')
                raise ValueError(msg)

        if not np.isclose(np.dot(self.major_axis_dir, self.major_plane_normal_dir), 0):
            msg = (f'`major_axis_dir` ({self.major_axis_dir}) and '
                   f'`major_plane_normal_dir` ({self.major_plane_normal_dir}) '
                   f'must be perpendicular.')
            raise ValueError(msg)

        self.major_axis_dir /= np.linalg.norm(self.major_axis_dir)
        self.major_plane_normal_dir /= np.linalg.norm(self.major_plane_normal_dir)

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
    def diameters(self):
        return tuple([self.major_axis_length] + list(self.minor_axes_lengths))

    @property
    def radii(self):
        return [i/2 for i in self.diameters]

    @property
    def volume(self):
        return (4/3) * np.pi * np.product(self.radii)

    @classmethod
    def from_diameters(cls, axis_sizes, centre, margins=None):
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
        new_diams = [i + j for i, j in zip(self.diameters, self.margins)]
        particle = Particle.from_diameters(new_diams, centre=self.centre)
        return particle

    @property
    def minor_axis_dir(self):
        """Unit vector of the third axis direction (i.e. after `major_axis_dir` and 
        `major_plane_normal_dir`"""
        xprod = np.cross(self.major_plane_normal_dir, self.major_axis_dir)
        minor_axis_dir = xprod / np.linalg.norm(xprod)
        return minor_axis_dir

    def displace(self, displacement):
        self.centre += displacement


class ParticleDistribution:

    def __init__(self, number=None, major_axis_length=None, minor_axis_ratios=None,
                 target_volume_fraction=None, major_axis_dir=None,
                 major_plane_normal_dir=None, margins=None, major_axis_length_stddev=None,
                 minor_axis_ratios_stddev=None, margins_stddev=None, random_seed=None,
                 label=None, _particles=None):

        if not _particles and sum(
            [i is not None for i in (number, major_axis_length, target_volume_fraction)]
        ) != 2:
            raise ValueError('Specify exactly two of `number`, `major_axis_length` and '
                             '`target_volume_fraction`.')

        if minor_axis_ratios is None:
            minor_axis_ratios = [1, 1]  # sphere
        if margins is None:
            margins = [0, 0, 0]

        if major_axis_dir is None and major_plane_normal_dir is not None:
            # find perpendicular major_axis_dir
            major_plane_normal_dir = np.array(major_plane_normal_dir, dtype=float)
            major_axis_dir = perpendicular_vectors(major_plane_normal_dir)
        elif major_axis_dir is not None and major_plane_normal_dir is None:
            # find perpendicular major_plane_normal_dir
            major_axis_dir = np.array(major_axis_dir, dtype=float)
            major_plane_normal_dir = perpendicular_vectors(major_axis_dir)
        else:
            if major_axis_dir is None:
                major_axis_dir = [1, 0, 0]
            if major_plane_normal_dir is None:
                major_plane_normal_dir = [0, 0, 1]

        self.label = label
        self.number = number
        self.major_axis_length = major_axis_length
        self.minor_axis_ratios = minor_axis_ratios
        self.target_volume_fraction = target_volume_fraction
        self.major_axis_dir = major_axis_dir
        self.major_plane_normal_dir = major_plane_normal_dir
        self.margins = margins
        self.random_seed = random_seed

        self.major_axis_length_stddev = major_axis_length_stddev
        self.minor_axis_ratios_stddev = minor_axis_ratios_stddev
        self.margins_stddev = margins_stddev

        self.particles = _particles or []

    def to_JSON_like(self):
        dct = {
            'label': self.label,
            'number': self.number,
            'major_axis_length': self.major_axis_length,
            'minor_axis_ratios': self.minor_axis_ratios,
            'target_volume_fraction': self.target_volume_fraction,
            'major_axis_dir': self.major_axis_dir,
            'major_plane_normal_dir': self.major_plane_normal_dir,
            'margins': self.margins,
            'random_seed': self.random_seed,
            'major_axis_length_stddev': self.major_axis_length_stddev,
            'minor_axis_ratios_stddev': self.minor_axis_ratios_stddev,
            'margins_stddev': self.margins_stddev,
            '_particles': [i.to_JSON_like() for i in self.particles]
        }
        return dct

    @classmethod
    def from_JSON_like(cls, dct):
        dct = copy.deepcopy(dct)
        particles = [Particle.from_JSON_like(i) for i in dct.pop('_particles', [])]
        obj = cls(_particles=particles, **dct)
        for particle in obj.particles:
            particle.distribution = obj
        return obj

    def calculate_missing_parameter(self, RVE_size):
        if self.number is None:
            self.calculate_number(RVE_size)
        elif self.major_axis_length is None:
            self.calculate_major_axis_length(RVE_size)
        elif self.target_volume_fraction is None:
            self.calculate_volume_fraction(RVE_size)
        else:
            return

    def calculate_number(self, RVE_size):

        if self.major_axis_length is None or self.target_volume_fraction is None:
            raise ValueError('Cannot calculate number of particles; `major_axis_length` '
                             'and `target_volume_fraction` must be set.')

        RVE_volume = np.product(RVE_size)
        particle_vol = (np.pi / 6) * (
            (self.major_axis_length ** 3) * np.product(self.minor_axis_ratios)
        )
        particle_vol_total = RVE_volume * self.target_volume_fraction
        number = int(np.round(particle_vol_total / particle_vol))

        self.number = number

    def calculate_major_axis_length(self, RVE_size):

        if self.number is None or self.target_volume_fraction is None:
            raise ValueError('Cannot calculate `major_axis_length`; `number` and '
                             '`target_volume_fraction` must be set.')

        RVE_volume = np.product(RVE_size)
        particle_vol_total = RVE_volume * self.target_volume_fraction
        particle_vol = particle_vol_total / self.number
        major_axis_length = (
            6 * particle_vol / (np.pi * np.product(self.minor_axis_ratios))
        ) ** (1 / 3)

        self.major_axis_length = major_axis_length

    def calculate_volume_fraction(self, RVE_size):

        if self.major_axis_length is None or self.number is None:
            raise ValueError('Cannot calculate `target_volume_fraction`; '
                             '`major_axis_length` and `number` must be set.')

        RVE_volume = np.product(RVE_size)
        particle_vol = (np.pi / 6) * (
            (self.major_axis_length ** 3) * np.product(self.minor_axis_ratios)
        )
        volume_fraction = self.number * particle_vol / RVE_volume

        self.target_volume_fraction = volume_fraction

    def set_particles(self, RVE_size):
        if not self.particles:
            self.calculate_missing_parameter(RVE_size)
            particles_info = generate_particle_distribution(
                number=self.number,
                major_axis_length=self.major_axis_length,
                minor_axis_ratios=self.minor_axis_ratios,
                major_axis_dir=self.major_axis_dir,
                major_plane_normal_dir=self.major_plane_normal_dir,
                margins=self.margins,
                major_axis_length_stddev=self.major_axis_length_stddev,
                minor_axis_ratios_stddev=self.minor_axis_ratios_stddev,
                margins_stddev=self.margins_stddev,
                random_seed=self.random_seed,
            )
            self.particles = [Particle(**i, distribution=self) for i in particles_info]


class ParticleRVE:

    def __init__(self, size, cells=None, material=None, particles=None, random_seed=None,
                 particle_distributions=None):
        """
        Parameters
        ----------
        size
        cells : list or ndarray of size three
            Number of cells in x, y, z directions. Specify exactly one of `cells` or
            `material`.
        material : nested non-ragged list or ndarray of dimension three
            Material index array (3D). Specify exactly one of `cells` or `material`.
        particles : list of (dict or Particle)
            Dict with the following keys:
                major_axis_size : number
                centre : list of number
                minor_axes_ratio : list of number, optional
        particle_distributions : list of (ParticleDistribution or dict)

        """

        # TODO: always save matrix material, so we can also remove particles.

        self.size = np.array(size)

        if cells is not None and material is not None:
            raise ValueError('Specify exactly one of `cells` and `material`, not both.')

        if material is not None:
            self._material = np.array(material)
            self.cells = material.shape

        else:
            self.cells = np.array(cells)
            self._material = np.zeros(self.cells, dtype=np.int64)

        self.particle_distributions = []
        self.particles = []  # lone particles not in a distribution

        self.grid_obj = self._generate_damask_grid_obj()

        self.matrix_material_IDs = list(np.unique(self._material))
        self.particle_material_IDs = {None: []}  # keyed by particle distribution label

        self.random_seed = random_seed or np.random.randint(0, 100_000)
        self.rng = np.random.default_rng(random_seed)

        self._add_particles(particles)

        for part_dist in particle_distributions or []:
            self.add_particle_distribution(part_dist)

    def to_JSON_like(self):
        """Get a JSON-compatible dict that can be reloaded with `from_json_like`."""
        dct = {
            'size': self.size,
            '_material': self._material,
            'particle_distributions': [i.to_JSON_like() for i in self.particle_distributions],
            'particles': [i.to_JSON_like() for i in self.particles],
            'random_seed': self.random_seed,
        }
        return dct

    @classmethod
    def from_JSON_like(cls, dct):
        dct = copy.deepcopy(dct)
        particles = [Particle.from_JSON_like(i) for i in dct['particles']]
        particle_distributions = [
            ParticleDistribution.from_JSON_like(i)
            for i in dct['particle_distributions']
        ]
        obj = cls(
            size=dct['size'],
            material=dct['_material'],
            particles=particles,
            particle_distributions=particle_distributions,
        )
        return obj

    @classmethod
    def from_voronoi_tessellation(cls, size, cells, seeds, particles=None,
                                  particle_distributions=None, random_seed=None):
        """
        Parameters
        ----------
        size : list or numpy.ndarray of shape (3,)
            Physical size of the grid in meter.
        cells : list or ndarray of size three
            Number of cells in x, y, z directions. Specify exactly one of `cells` or
            `material`.
        seeds : numpy.ndarray of shape (N, 3)
            Position of the seed points.        
        particles : list of (dict or Particle)
            Dict with the following keys:
                major_axis_size : number
                centre : list of number
                minor_axes_ratio : list of number, optional
        particle_distributions : list of (dict or ParticleDistribution)

        """
        from damask import Grid

        grid_obj = Grid.from_Voronoi_tessellation(cells, size, seeds)

        return cls(
            size=size,
            material=grid_obj.material,
            particles=particles,
            particle_distributions=particle_distributions,
            random_seed=random_seed,
        )

    def _add_particles(self, particles):
        for particle in particles or []:
            if not isinstance(particle, Particle):
                particle = Particle(**particle)
            self._add_particle(particle)

    @property
    def all_particles(self):
        return self.particles + [
            j for i in self.particle_distributions for j in i.particles
        ]

    @property
    def material(self):
        return self.grid_obj.material

    @property
    def num_voxels(self):
        return np.product(self.cells)

    @property
    def volume(self):
        return np.product(self.size)

    @property
    def matrix_voxel_volume_fraction(self):
        """Volume fraction of matrix based on voxel assignment."""
        return np.sum(self.matrix_voxels) / self.num_voxels

    @property
    def matrix_volume_fraction(self):
        """Volume fraction of matrix for infinite grid size."""
        return 1 - self.total_particle_volume_fraction

    @property
    def total_particle_voxel_volume_fraction(self):
        """Volume fraction of all particles based on voxel assignment."""
        return 1 - self.matrix_voxel_volume_fraction

    @property
    def total_particle_volume_fraction(self):
        """Volume fraction of all particles for infinite grid size."""
        particle_vols = np.sum([i.volume for i in self.all_particles])
        return particle_vols / self.volume

    def get_particle_distribution_voxel_volume_fraction(self, dist_label):
        """Volume fraction of particles of a given distribution based on voxel
        assignment."""
        return np.sum(self.get_particle_distribution_voxels(dist_label)) / self.num_voxels

    def get_particle_distribution_volume_fraction(self, dist_label):
        """Volume fraction of particles of a given distribution for infinite grid size."""
        particle_vols = np.sum(
            [i.volume for i in self.get_particle_distribution(dist_label).particles]
        )
        return particle_vols / self.volume

    def get_particle_distribution(self, dist_label):
        for i in self.particle_distributions:
            if i.label == dist_label:
                return i

        raise ValueError(f'No particle distribution: "{dist_label}"')

    @property
    def num_distributions(self):
        return len(self.particle_distributions)

    def add_particle_distribution(self, particle_distribution, max_iter=1_000):

        if not isinstance(particle_distribution, ParticleDistribution):
            particle_distribution = ParticleDistribution(**particle_distribution)

        find_good_position = False
        if not particle_distribution.particles:
            # if particles have already been set, assume positions are good:
            particle_distribution.set_particles(self.size)
            find_good_position = True

        if particle_distribution.label is None:
            particle_distribution.label = self.num_distributions

        label = particle_distribution.label
        self.particle_material_IDs.update({label: []})
        self.particle_distributions.append(particle_distribution)
        self.add_particles(
            particle_distribution.particles,
            find_good_position=find_good_position,
            max_iter=max_iter,
            dist_label=label,
        )

    def _get_random_centre(self):
        centre = self.rng.random(3,)
        centre[0] *= self.size[0]
        centre[1] *= self.size[1]
        centre[2] *= self.size[2]
        return centre

    def _generate_damask_grid_obj(self):
        from damask import Grid
        grid_obj = Grid(material=self._material, size=self.size)
        return grid_obj

    def _add_particle(self, particle, dist_label=None):
        from damask import Rotation
        rot_mat = np.array([
            particle.major_axis_dir,
            particle.minor_axis_dir,
            particle.major_plane_normal_dir,
        ]).T
        grid_obj = self.grid_obj.add_primitive(
            dimension=particle.diameters,
            center=particle.centre,
            exponent=1,
            R=Rotation.from_matrix(rot_mat),
        )
        self.particle_material_IDs[dist_label].append(grid_obj.material.max())
        self.grid_obj = grid_obj

        if dist_label is None:
            # Particle does not belong to a distribution:
            self.particles.append(particle)

    @property
    def matrix_voxels(self):
        """Get a bool array that shows where matrix voxels are."""
        return np.logical_or.reduce([
            self.material == i for i in self.matrix_material_IDs
        ])

    @property
    def particle_voxels(self):
        """Get a bool array that shows where particle voxels are."""
        return np.logical_not(self.matrix_voxels)

    def get_particle_distribution_voxels(self, dist_label):
        """Get a bool array that shows where particles voxels are that belong to a given 
        distribution."""
        return np.logical_or.reduce([
            self.material == i for i in self.particle_material_IDs[dist_label]
        ])

    @property
    def num_voxels(self):
        return np.product(self.cells)

    def add_particle(self, particle, find_good_position=False, max_iter=1_000,
                     dist_label=None):
        """Add a particle to the RVE."""

        if not isinstance(particle, Particle):
            particle = Particle(**particle)

        if not find_good_position:
            # Add without searching for a good position:
            print(f'Using existing centre.')
            self._add_particle(particle, dist_label=dist_label)

        else:
            # Iterate on position to maintain margins between particles

            # Set a random position:
            particle.centre = self._get_random_centre()

            # Generate an RVE with a single (expanded-by-margins) particle in it:
            single_particle_RVE = ParticleRVE(
                self.size,
                self.cells,
                particles=[particle.margined_particle],
            )

            # Find overlap voxels with existing particles:
            overlap_bool = np.logical_and(
                self.particle_voxels,
                single_particle_RVE.material != 0
            )
            single_particle_RVE_voxels = single_particle_RVE.material != 0
            overlap_fraction = np.sum(overlap_bool) / np.sum(single_particle_RVE_voxels)
            count = 0
            while np.any(overlap_bool):
                if count > max_iter:
                    print(f'Cannot find suitable position for particle in {max_iter} '
                          f'iterations. Skipping.')
                    return

                # Try a new position:
                particle.centre = self._get_random_centre()
                single_particle_RVE = ParticleRVE(
                    self.size,
                    self.cells,
                    particles=[particle.margined_particle],
                )
                overlap_bool = np.logical_and(
                    self.particle_voxels,
                    single_particle_RVE.material != 0
                )
                single_particle_RVE_voxels = single_particle_RVE.material != 0
                overlap_fraction = np.sum(overlap_bool) / \
                    np.sum(single_particle_RVE_voxels)
                count += 1

            print(f'Found suitable particle centre in {count} iterations.')

            self._add_particle(particle, dist_label=dist_label)

    def add_particles(self, particles, find_good_position=False, max_iter=1_000,
                      dist_label=None):
        print(f'Adding {len(particles)} particles...')
        for idx, particle in enumerate(particles):
            print(f'Adding particle {idx}... ', end='')
            self.add_particle(
                particle,
                find_good_position,
                max_iter=max_iter,
                dist_label=dist_label,
            )

    def save(self, *args, **kwargs):
        """Call save on the DAMASK Grid object."""
        self.grid_obj.save(*args, **kwargs)

    def write_VTR_particle_history(self, directory):
        """Generate a series of VTR files showing the insertion of the particles."""

        from damask import Rotation

        base_grid = self._generate_damask_grid_obj()
        num_inserts = sum([len(particle.centre_history)
                           for particle in self.all_particles])
        zero_pad_len = len(str(num_inserts))
        count = 0
        base_grid.save(f'{directory}/particle_RVE_{count:0{zero_pad_len}}.vtr')
        for particle in self.all_particles:
            rot_mat = np.array([
                particle.major_axis_dir,
                particle.minor_axis_dir,
                particle.major_plane_normal_dir,
            ]).T
            for historic_centre in particle.centre_history:
                count += 1
                grid = base_grid.add_primitive(
                    dimension=particle.diameters,
                    center=historic_centre,
                    exponent=1,
                    R=Rotation.from_matrix(rot_mat),
                )
                grid.save(f'{directory}/particle_RVE_{count:0{zero_pad_len}}.vtr')
            base_grid = grid

    def show_slice(self):
        from plotly import graph_objects
        data = [
            {
                'type': 'heatmap',
                'z': self.grid_obj.material[int(self.cells[0]/2)],
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
        coords = get_coordinate_grid(self.size, self.cells)
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
