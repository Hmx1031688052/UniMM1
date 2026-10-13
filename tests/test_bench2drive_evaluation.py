import unittest

import numpy as np

from bench2drive_evaluate import displacement_errors, choose_windows


class EvaluationTest(unittest.TestCase):
    def test_masks_and_oracle_errors(self):
        target = np.zeros((3, 4, 2))
        prediction = np.zeros((2, 3, 4, 2))
        prediction[0, :, :, 0] = 1
        prediction[1, :, :, 0] = 3
        valid = np.array([[True, True, True, True], [True, True, False, False],
                          [False, False, False, False]])
        prediction[:, 1, 2:] = 10000
        errors = displacement_errors(prediction, target, valid)
        np.testing.assert_allclose(errors['meanADE_m'][:2], [2, 2])
        np.testing.assert_allclose(errors['minADE_m'][:2], [1, 1])
        self.assertEqual(errors['meanFDE_m'][0], 2)
        self.assertTrue(np.isnan(errors['meanFDE_m'][1:]).all())
        self.assertTrue(np.isnan(errors['meanADE_m'][2]))

    def test_subset_spreads_over_clips(self):
        infos = [{'scenario_id': f'{clip}__{i:05d}'} for clip in ['a', 'b', 'c'] for i in range(5)]
        indices = choose_windows(infos, 3, 2025)
        self.assertEqual(len({infos[i]['scenario_id'].split('__')[0] for i in indices}), 3)
        self.assertEqual(indices, choose_windows(infos, 3, 2025))
        self.assertEqual(len(choose_windows(infos, 0, 2025)), 15)


if __name__ == '__main__':
    unittest.main()
