"""Procedural finite-search maxima are not two-sided localizations."""
import unittest
from fractions import Fraction as F
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from ez2rc.v2.difference_constraints import ConstraintConflict
from ez2rc.v2.motion import STATUS
from ez2rc.v2.observed_geometry import (LEGACY_LOCALIZATION_SCOPE, LOCALIZATION_SCOPE,
                                       observed_geometry, search_censored)


def translated_observations():
    """A head, an unrelated response, and a landmark beyond an observation gap."""
    chunk = {
        'frame_id': np.arange(3),
        'candidate_index': np.array([[0,0,80], [1,0,90], [1,1,40], [2,1,60], [2,2,30]]),
        'positions': np.array([[[0,80],[0,80],[0,82]], [[0,90],[0,90],[0,100]],
                               [[0,40]]*3, [[0,60]]*3, [[0,30]]*3]),
        'scores': np.full((5,3), .99), 'flags': np.zeros((5,3), dtype=int),
    }
    clock = {'frame_id':np.arange(3), 'status':np.array([STATUS['origin'],STATUS['observed'],STATUS['observed']]),
             'reference_frame':np.array([-1,0,1]), 'scroll_px':np.array([0.,10.,30.]),
             'repeat_confirmed':np.zeros(3,dtype=bool), 'repeat_group':np.arange(3)}
    links = np.array([[1,0,0,1,80,90,10,.99,0], [1,0,0,2,80,90,18,.99,0],
                      [2,1,1,0,40,60,20,.99,0]])
    profile = {'judge_y':180, 'lanes':[{'roi':[0,0,8,200]} for _ in range(3)]}
    observations = SimpleNamespace(id='observations', chunks=lambda:iter([chunk]), manifest={
        'metadata':{'profile':profile}, 'key':{'parameters':{'algorithm':{'search_radius_px':12}}}})
    movement = SimpleNamespace(id='movement', chunks=lambda:iter([{'links':links}]),
        manifest={'key':{'parameters':{'min_similarity':.9}}})
    decoded = SimpleNamespace(id='decoded')
    selections = [
        {'lane':0, 'endpoints':[{'id':'head', 'kind':'head', 'observation_ids':['f0:l0:y80','f1:l0:y90']}],
         'candidates':[{'selected':True,'start':'head','end':None}]},
        {'lane':2, 'endpoints':[{'id':'distant', 'kind':'tap', 'observation_ids':['f2:l2:y30']}],
         'candidates':[{'selected':True,'start':'distant','end':None}]},
    ]
    return observations, movement, decoded, selections, clock, chunk


class ObservedGeometryTest(unittest.TestCase):
    def test_repeated_texture_has_high_correlation_without_a_position_bound(self):
        failures = 0
        for seed in range(32):
            rng = np.random.default_rng(seed)
            stripe = np.array([35,65,180,235,180,65,35,65,180,235,180,65,35],np.float32)
            body = np.tile(stripe,(180,1))
            cap = body[:12].copy()
            cap[-4:] = np.array([240,240,210,80],np.float32)[:,None]
            alias = body[:12] + rng.normal(0,.4,cap.shape).astype(np.float32)
            results = [[],[]]
            for end in (93,128):
                scene = np.zeros_like(body)
                scene[:end] = body[:end]; scene[end-12:end] = cap
                scene += rng.normal(0,.6,scene.shape).astype(np.float32)
                lower,upper = end-42,end-4
                for i,template in enumerate((cap,alias)):
                    scores = cv2.matchTemplate(scene[lower:upper+12],template,cv2.TM_CCOEFF_NORMED).ravel()
                    at = int(scores.argmax())
                    results[i].append((lower+at,float(scores[at]),at not in (0,len(scores)-1)))
            self.assertEqual(results[0][1][0]-results[0][0][0],35)
            if all(row[2] and row[1]>.99 for row in results[1]):
                failures += abs(results[1][1][0]-results[1][0][0]-35)>4
        self.assertGreater(failures,0)

    def project_scene(self, *, legacy=False, conflict=False, effect_bridge=False, review=None, error=0):
        observations,movement,decoded,selections,clock,chunk = translated_observations()
        if effect_bridge:chunk['flags'][0,1] |= 2
        before = {k:v.copy() for k,v in chunk.items()}
        anchors = {'zero':[{'frame_id':0,'y_px':160}], 'one':[{'frame_id':0,'y_px':60}]}
        if conflict:
            anchors['zero'].append({'frame_id':1,'y_px':180})
        with patch('ez2rc.v2.observed_geometry.timeline',return_value=clock), \
             patch('ez2rc.v2.observed_geometry.lanes',return_value=selections), \
             patch('ez2rc.v2.observed_geometry.template_shapes',return_value=[{k:(3,8) for k in ('tap','head','tail')}]*3):
            result = observed_geometry(observations,movement,decoded,{'zero':F(0),'one':F(1)},anchors,error,
                localization_scope=LEGACY_LOCALIZATION_SCOPE if legacy else LOCALIZATION_SCOPE,
                reviewed_localizations=[review] if review else ())
        for key,value in before.items():np.testing.assert_array_equal(chunk[key],value)
        return result

    def test_kind_scope_ignores_unselected_alias_without_mutating_raw_evidence(self):
        model,nodes,diagnostics = self.project_scene()
        self.assertEqual(model.status,'conditional')
        self.assertEqual(model.scale,(100,100))
        self.assertEqual(diagnostics['raw_observations'],3)
        self.assertEqual(diagnostics['excluded_observations']['outside_selected_endpoint_kind'],4)
        self.assertEqual(diagnostics['localization_scope'],LOCALIZATION_SCOPE)
        self.assertFalse(diagnostics['verified'])
        self.assertEqual(model.project(nodes['head']),(F(4,5),F(4,5)))
        # High-scoring aliases cannot create a bridge across an unmeasured gap.
        self.assertIsNone(model.project(nodes['distant']))
        with self.assertRaises(ConstraintConflict):self.project_scene(legacy=True)

    def test_conflicting_selected_localizations_are_not_removed_to_obtain_a_fit(self):
        with self.assertRaises(ConstraintConflict):self.project_scene(conflict=True)

    def test_explicit_visible_measurement_bridges_only_its_reviewed_observation(self):
        model,nodes,_ = self.project_scene(effect_bridge=True,error=1)
        self.assertIsNone(model.project(nodes['head']))
        review={'endpoint_id':'head','observation_id':'f0:l0:y80','frame_id':0,
                'template_top_interval_px':[79,81],'visibility':'fully_visible_unoccluded',
                'evidence_ids':['procedural-renderer']}
        model,nodes,diagnostics = self.project_scene(effect_bridge=True,review=review,error=1)
        lower,upper=model.project(nodes['head'])
        self.assertLessEqual(lower,F(4,5));self.assertGreaterEqual(upper,F(4,5))
        self.assertIsNone(model.project(nodes['distant']))
        self.assertEqual(diagnostics['reviewed_localizations'],[review])
        self.assertFalse(diagnostics['verified'])
        with self.assertRaisesRegex(ValueError,'understates'):
            self.project_scene(effect_bridge=True,error=1,
                               review={**review,'template_top_interval_px':[80,80]})

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
