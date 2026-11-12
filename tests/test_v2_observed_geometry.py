"""Procedural finite-search maxima are not two-sided localizations."""
import unittest

import cv2
import numpy as np

from ez2rc.v2.observed_geometry import search_censored


class ObservedGeometryTest(unittest.TestCase):
    def test_fully_visible_pattern_can_be_censored_by_the_search_window(self):
        x,y=np.meshgrid(np.arange(7),np.arange(80))
        image=np.exp(-((x-3)**2/4+(y-24)**2/16)).astype('float32')
        template=image[20:29].copy()
        for seed in (15,20,25,30):
            lower,upper=seed-4,seed+4
            scores=cv2.matchTemplate(image[lower:upper+9],template,cv2.TM_CCOEFF_NORMED)
            _,_,_,location=cv2.minMaxLoc(scores)
            measured=lower+location[1]
            censored=search_censored(seed,measured,[0,0,7,80],9,4)
            self.assertEqual(censored,seed!=20)
            if not censored:self.assertEqual(measured,20)
        # A bottom-clamped search and a translated viewport have the same rule.
        self.assertTrue(search_censored(75,71,[0,0,7,80],9,4))
        self.assertTrue(search_censored(25,121,[0,100,7,180],9,4))
        self.assertFalse(search_censored(20,120,[0,100,7,180],9,4))
