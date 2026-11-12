from fractions import Fraction as F
import unittest

from ez2rc.v2.difference_constraints import DifferenceGraph
from ez2rc.v2.rhythm import RhythmGrid
from ez2rc.v2.rhythm_refinement import refine_grid
from ez2rc.v2.spatial_constraints import ConstantSpacingMap


def point(identity,status='resolved'):
    return {'id':identity,'node':identity,'offset':(F(0),F(0)),'beat_shift':F(0),
            'candidate_id':identity,'side':'start','pixel_status':status}


class RhythmRefinementTest(unittest.TestCase):
    def model(self):
        graph=DifferenceGraph()
        graph.bound('a','b',36,44,'scale')
        graph.bound('a','p',10,10,'first observation')
        graph.bound('a','q',14,18,'second observation')
        return ConstantSpacingMap(graph,{'a':F(0),'b':F(4)},['p','q'])

    def test_unique_grid_label_tightens_shared_scale_before_next_decision(self):
        model,proof=refine_grid(self.model(),[point('p'),point('q')],RhythmGrid([2]))
        self.assertEqual(model.status,'conditional')
        self.assertEqual(model.scale,(F(10),F(10)))
        self.assertEqual(proof['anchored_endpoint_ids'],['p','q'])
        self.assertEqual(len(proof['iterations']),2)
        self.assertEqual(proof['iterations'][1]['added_unique_labels'][0]['beat'],[3,2])
        self.assertEqual(proof['accepted_note_count'],0)

    def test_ambiguous_physical_interpretation_does_not_calibrate_others(self):
        model,proof=refine_grid(self.model(),[point('p','ambiguous'),point('q')],RhythmGrid([2]))
        self.assertEqual(proof['anchored_endpoint_ids'],[])
        self.assertEqual(model.scale,(F(9),F(11)))

    def test_individually_unique_labels_can_conflict_jointly(self):
        graph=DifferenceGraph();graph.bound('a','b',36,44,'scale')
        graph.bound('a','p',F(99,10),F(101,10),'first')
        graph.bound('a','r',F(215,10),F(219,10),'second')
        model=ConstantSpacingMap(graph,{'a':F(0),'b':F(4)},['p','r'])
        model,proof=refine_grid(model,[point('p'),point('r')],RhythmGrid([1]))
        self.assertEqual(model.status,'conflict')
        self.assertEqual(proof['accepted_note_count'],0)

    def test_simultaneous_notes_share_a_label_without_duplicate_anchor_failure(self):
        model=self.model();model.graph.bound('p','simultaneous',0,0,'same rendered position')
        model=ConstantSpacingMap(model.graph,model.anchors,['p','simultaneous'])
        model,proof=refine_grid(model,[point('p'),point('simultaneous')],RhythmGrid([1]))
        self.assertEqual(model.status,'conditional')
        self.assertEqual(model.scale,(F(10),F(10)))

    def test_absent_grid_label_is_a_conflict_not_a_silently_skipped_seed(self):
        graph=DifferenceGraph();graph.bound('a','b',40,40,'scale')
        graph.bound('a','p',12,13,'off-grid observation')
        model=ConstantSpacingMap(graph,{'a':F(0),'b':F(4)},['p'])
        model,proof=refine_grid(model,[point('p')],RhythmGrid([1]))
        self.assertEqual(proof['status'],'conflict')
        self.assertEqual(proof['conflicts'][0]['endpoint_ids'],['p'])
        self.assertEqual(proof['anchored_endpoint_ids'],[])
        self.assertEqual(model.project('p'),(F(6,5),F(13,10)))

    def test_nonpositive_duration_conflicts_before_seeding(self):
        graph=DifferenceGraph();graph.bound('a','b',40,40,'scale')
        graph.bound('a','p',20,20,'head');graph.bound('a','q',10,10,'tail')
        model=ConstantSpacingMap(graph,{'a':F(0),'b':F(4)},['p','q'])
        points=[dict(point('p'),candidate_id='ln'),dict(point('q'),candidate_id='ln',side='end')]
        _,proof=refine_grid(model,points,RhythmGrid([1]))
        self.assertEqual(proof['conflicts'][0]['reason'],'no_positive_grid_duration')
        self.assertEqual(proof['anchored_endpoint_ids'],[])
