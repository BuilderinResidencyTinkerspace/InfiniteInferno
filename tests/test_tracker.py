import unittest

import numpy as np

from src.controller import FollowController
from src.gestures import GestureEvent, GestureRecognizer
from src.postprocess import Detection
from src.tracker import PersonTracker, Track


def detection(box):
    return Detection(box, 0.9)


def pose_track(track_id=1, left_up=False, right_up=False):
    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.9
    points[0, :2] = [100, 80]
    points[5, :2] = [80, 120]
    points[6, :2] = [120, 120]
    points[9, :2] = [70, 90 if left_up else 160]
    points[10, :2] = [130, 90 if right_up else 160]
    return Track(track_id, (50, 50, 150, 250), 0.9, points, hits=3, confirmed=True)


class TrackerTests(unittest.TestCase):
    def test_id_survives_motion_and_brief_occlusion(self):
        tracker = PersonTracker(max_missed=2, min_hits=1)
        first = tracker.update([detection((10, 10, 110, 210))], (640, 480))[0]
        moved = tracker.update([detection((20, 12, 120, 212))], (640, 480))[0]
        self.assertEqual(moved.track_id, first.track_id)
        self.assertEqual(tracker.update([], (640, 480)), [])
        returned = tracker.update([detection((25, 15, 125, 215))], (640, 480))[0]
        self.assertEqual(returned.track_id, first.track_id)

    def test_two_people_get_different_ids(self):
        tracker = PersonTracker(min_hits=1)
        tracks = tracker.update([detection((10, 10, 100, 200)), detection((400, 10, 500, 200))], (640, 480))
        self.assertEqual({track.track_id for track in tracks}, {1, 2})


class GestureTests(unittest.TestCase):
    def test_gesture_requires_hold_and_emits_for_correct_person(self):
        recognizer = GestureRecognizer(hold_frames=3, cooldown_seconds=2)
        track = pose_track(track_id=7, right_up=True)
        self.assertEqual(recognizer.update([track], now=0), [])
        self.assertEqual(recognizer.update([track], now=0.1), [])
        events = recognizer.update([track], now=0.2)
        self.assertEqual(events, [GestureEvent(7, "right_hand_up", 0.2)])
        self.assertEqual(recognizer.update([track], now=0.3), [])

    def test_both_hands_has_priority(self):
        recognizer = GestureRecognizer(hold_frames=1)
        event = recognizer.update([pose_track(left_up=True, right_up=True)], now=1)[0]
        self.assertEqual(event.name, "both_hands_up")


class ControllerTests(unittest.TestCase):
    def test_only_gesturing_person_becomes_target(self):
        controller = FollowController(gesture_actions={"right_hand_up": "start_follow"})
        target = pose_track(track_id=2)
        event = GestureEvent(2, "right_hand_up", 0)
        intent = controller.update([pose_track(1), target], [event], (640, 480), now=0)
        self.assertEqual(intent.action, "follow")
        self.assertEqual(intent.target_id, 2)

    def test_target_loss_commands_hover(self):
        controller = FollowController(gesture_actions={"right_hand_up": "start_follow"})
        controller.update([pose_track(1)], [GestureEvent(1, "right_hand_up", 0)], (640, 480), now=0)
        intent = controller.update([], [], (640, 480), now=1)
        self.assertEqual(intent.action, "hover")
        self.assertEqual(intent.reason, "target lost")

    def test_land_gesture_stops_following(self):
        controller = FollowController(gesture_actions={"both_hands_up": "land"})
        intent = controller.update([pose_track(3)], [GestureEvent(3, "both_hands_up", 0)], (640, 480), now=0)
        self.assertEqual(intent.action, "land")
        self.assertFalse(controller.following)

    def test_bystander_cannot_land_after_target_is_locked(self):
        controller = FollowController(gesture_actions={"both_hands_up": "land"})
        controller.start_follow(3)
        intent = controller.update([pose_track(3)], [GestureEvent(9, "both_hands_up", 0)], (640, 480), now=0)
        self.assertEqual(intent.action, "follow")
        self.assertTrue(controller.following)


if __name__ == "__main__":
    unittest.main()
