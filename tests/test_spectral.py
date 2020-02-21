"""`test_spectral.py`

Tests `read_spectral_stdout` and `read_spectral_stderr`.

"""

from unittest import TestCase
from pathlib import Path

import numpy as np

from damask_parse.readers import parse_increment, parse_increment_iteration


class SpectralStdOutTestCase(TestCase):

    def test_parse_increment(self):

        inc_str = """
            Time 1.00000E+01s: Increment 10/500-1/1 of load case 1/1
            Increment 10/500-1/1 @ Iteration 001â‰¤000â‰¤250

            deformation gradient aim       =
            0.9958367    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                -0.0190        -0.0320        -0.0473 
                -0.0322        26.3463        -0.3616 
                -0.0478        -0.3634        55.3017 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         9.37 (2.59E+05 / m, tol =  2.77E+04)
            error stress BC  =         0.03 (1.90E+04 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤001â‰¤250

            deformation gradient aim       =
            0.9958370    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0086        -0.0322        -0.0475 
                -0.0324        26.3578        -0.3615 
                -0.0479        -0.3633        55.3290 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0091        -0.0322        -0.0475 
                -0.0324        26.3580        -0.3615 
                -0.0479        -0.3633        55.3295 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         5.40 (1.49E+05 / m, tol =  2.77E+04)
            error stress BC  =         0.02 (9.10E+03 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤002â‰¤250

            deformation gradient aim       =
            0.9958368    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                -0.0028        -0.0326        -0.0480 
                -0.0328        26.3498        -0.3615 
                -0.0485        -0.3633        55.3213 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0006        -0.0328        -0.0482 
                -0.0329        26.3505        -0.3614 
                -0.0486        -0.3633        55.3255 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         3.47 (9.60E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (6.42E+02 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤003â‰¤250

            deformation gradient aim       =
            0.9958368    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0010        -0.0326        -0.0484 
                -0.0327        26.3503        -0.3614 
                -0.0489        -0.3632        55.3255 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                -0.0016        -0.0327        -0.0488 
                -0.0328        26.3477        -0.3614 
                -0.0492        -0.3632        55.3243 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         2.54 (7.03E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (1.55E+03 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤004â‰¤250

            deformation gradient aim       =
            0.9958368    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0012        -0.0328        -0.0489 
                -0.0329        26.3499        -0.3614 
                -0.0493        -0.3632        55.3269 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0017        -0.0328        -0.0491 
                -0.0330        26.3498        -0.3614 
                -0.0496        -0.3632        55.3278 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         2.09 (5.79E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (1.69E+03 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤005â‰¤250

            deformation gradient aim       =
            0.9958368    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0002        -0.0329        -0.0492 
                -0.0331        26.3489        -0.3613 
                -0.0497        -0.3631        55.3266 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0009        -0.0331        -0.0494 
                -0.0332        26.3494        -0.3613 
                -0.0499        -0.3631        55.3277 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         1.78 (4.91E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (9.15E+02 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤006â‰¤250

            deformation gradient aim       =
            0.9958367    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0005        -0.0330        -0.0494 
                -0.0332        26.3492        -0.3613 
                -0.0499        -0.3631        55.3273 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                -0.0002        -0.0331        -0.0495 
                -0.0333        26.3488        -0.3612 
                -0.0500        -0.3630        55.3270 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         1.50 (4.14E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (1.81E+02 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤007â‰¤250

            deformation gradient aim       =
            0.9958367    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0007        -0.0331        -0.0495 
                -0.0333        26.3495        -0.3612 
                -0.0499        -0.3630        55.3279 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0007        -0.0332        -0.0495 
                -0.0333        26.3497        -0.3611 
                -0.0500        -0.3629        55.3281 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         1.27 (3.51E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (7.20E+02 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤008â‰¤250

            deformation gradient aim       =
            0.9958367    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0004        -0.0332        -0.0495 
                -0.0333        26.3495        -0.3611 
                -0.0499        -0.3629        55.3279 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0008        -0.0332        -0.0494 
                -0.0334        26.3499        -0.3611 
                -0.0499        -0.3629        55.3284 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         1.08 (2.99E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (7.68E+02 Pa,  tol =  5.53E+05)

            ===========================================================================
            Increment 10/500-1/1 @ Iteration 001â‰¤009â‰¤250

            deformation gradient aim       =
            0.9958367    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0050000 

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0004        -0.0332        -0.0494 
                -0.0333        26.3497        -0.3611 
                -0.0499        -0.3629        55.3281 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                    0.0002        -0.0332        -0.0493 
                -0.0333        26.3496        -0.3610 
                -0.0498        -0.3628        55.3281 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =         0.91 (2.52E+04 / m, tol =  2.77E+04)
            error stress BC  =         0.00 (1.95E+02 Pa,  tol =  5.53E+05)

            ===========================================================================

            increment 10 donverged

            ... writing results to file ......................................

            ┌─────────────────────────────────────────────────────────────────────┐
            │                        warning                                      │
            │                        850                                          │
            ├─────────────────────────────────────────────────────────────────────┤
            │ max number of cut back exceeded, terminating                        │
            │                                                                     │
            └─────────────────────────────────────────────────────────────────────┘

        
        """

        parse_increment(inc_str)
        # TODO: test some things!

    def test_parse_increment_iteration(self):

        inc_iter_str = """
            Time 1.00000E+00s: Increment 1/500-1/1 of load case 1/1
            Increment 1/500-1/1 @ Iteration 001â‰¤000â‰¤250

            deformation gradient aim       =
            1.0000000    0.0000000    0.0000000 
            0.0000000    1.0000000    0.0000000 
            0.0000000    0.0000000    1.0005000     

            ... evaluating constitutive response ......................................

            Piola--Kirchhoff stress       / MPa =
                23.6511        -0.0171        -0.0231 
                -0.0171        23.5818         0.0283 
                -0.0231         0.0284        44.2691 

            ... calculating divergence ................................................

            ... doing gamma convolution ...............................................

            ... reporting .............................................................

            error divergence =       676.79 (1.50E+07 / m, tol =  2.21E+04)
            error stress BC  =        53.43 (2.37E+07 Pa,  tol =  4.43E+05)
        """

        out = parse_increment_iteration(inc_iter_str)

        self.assertTrue(
            np.allclose(
                out['deformation_gradient_aim'],
                np.array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1.0005],
                ])
            )
        )
        self.assertTrue(
            np.allclose(
                out['piola_kirchhoff_stress'],
                np.array([
                    [23.6511, -0.0171, -0.0231],
                    [-0.0171, 23.5818, 0.0283],
                    [-0.0231, 0.0284, 44.2691],
                ])
            )
        )
        self.assertTrue(np.isclose(out['error_divergence']['value'], 1.5e7))
        self.assertTrue(np.isclose(out['error_divergence']['tol'], 2.21e4))
        self.assertTrue(np.isclose(out['error_divergence']['relative'], 676.79))

        self.assertTrue(np.isclose(out['error_stress_BC']['value'], 2.37e7))
        self.assertTrue(np.isclose(out['error_stress_BC']['tol'], 4.43e5))
        self.assertTrue(np.isclose(out['error_stress_BC']['relative'], 53.43))
