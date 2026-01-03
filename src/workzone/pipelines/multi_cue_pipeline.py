"""
Complete Multi-Cue Persistence Pipeline
Phase 1.1: Integration of all components
"""

from typing import Optional, List, Dict
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
import cv2
import pandas as pd
from tqdm import tqdm

from ..detection import CueClassifier, FrameCues
from ..temporal import PersistenceTracker, WorkZoneStateMachine, WorkZoneState
from ..fusion import MultiCueGate


@dataclass
class FrameResult:
    """Complete result for one frame."""
    frame_id: int
    timestamp: float
    
    # Cue detections
    channelization_present: bool
    channelization_count: int
    channelization_persistence: float
    
    signage_present: bool
    signage_count: int
    signage_persistence: float
    
    personnel_present: bool
    personnel_count: int
    personnel_persistence: float
    
    equipment_present: bool
    equipment_count: int
    equipment_persistence: float
    
    infrastructure_present: bool
    infrastructure_count: int
    infrastructure_persistence: float
    
    # Multi-cue decision
    multi_cue_pass: bool
    num_cues_sustained: int
    sustained_cues: str  # Comma-separated
    
    # State machine
    state: str
    state_confidence: float


class MultiCuePipeline:
    """
    Complete pipeline integrating all Phase 1.1 components.
    
    Usage:
        pipeline = MultiCuePipeline(yolo_model, config_path)
        results = pipeline.process_video(video_path)
        pipeline.save_results(results, output_csv)
    """
    
    def __init__(
        self,
        yolo_model,
        config_path: Optional[str] = None
    ):
        """
        Initialize pipeline with YOLO model.
        
        Args:
            yolo_model: Loaded YOLO model (ultralytics)
            config_path: Path to multi_cue_config.yaml
        """
        self.yolo_model = yolo_model
        
        # Initialize components
        self.cue_classifier = CueClassifier(config_path)
        self.persistence_tracker = PersistenceTracker(config_path)
        self.multi_cue_gate = MultiCueGate(config_path)
        self.state_machine = WorkZoneStateMachine(config_path)
        
        # Results storage
        self.frame_results: List[FrameResult] = []
    
    def process_frame(
        self,
        frame,
        frame_id: int,
        timestamp: float
    ) -> FrameResult:
        """
        Process a single frame through the complete pipeline.
        
        Args:
            frame: Video frame (numpy array)
            frame_id: Frame number
            timestamp: Timestamp in seconds
            
        Returns:
            FrameResult with all pipeline outputs
        """
        # 1. YOLO detection
        yolo_results = self.yolo_model(frame, verbose=False)[0]
        
        # 2. Cue classification
        frame_cues = self.cue_classifier.classify_detections(
            yolo_results,
            frame_id,
            timestamp
        )
        
        # 3. Persistence tracking
        persistence_states = self.persistence_tracker.update(frame_cues)
        
        # 4. Multi-cue gate
        multi_cue_decision = self.multi_cue_gate.evaluate(
            frame_cues,
            persistence_states
        )
        
        # 5. State machine
        current_state = self.state_machine.update(
            frame_cues,
            persistence_states,
            multi_cue_decision
        )
        
        # 6. Package results
        result = FrameResult(
            frame_id=frame_id,
            timestamp=timestamp,
            
            # Cue presence & persistence
            channelization_present=frame_cues.cue_groups['CHANNELIZATION']['present'],
            channelization_count=frame_cues.cue_groups['CHANNELIZATION']['count'],
            channelization_persistence=persistence_states['CHANNELIZATION'].persistence_score,
            
            signage_present=frame_cues.cue_groups['SIGNAGE']['present'],
            signage_count=frame_cues.cue_groups['SIGNAGE']['count'],
            signage_persistence=persistence_states['SIGNAGE'].persistence_score,
            
            personnel_present=frame_cues.cue_groups['PERSONNEL']['present'],
            personnel_count=frame_cues.cue_groups['PERSONNEL']['count'],
            personnel_persistence=persistence_states['PERSONNEL'].persistence_score,
            
            equipment_present=frame_cues.cue_groups['EQUIPMENT']['present'],
            equipment_count=frame_cues.cue_groups['EQUIPMENT']['count'],
            equipment_persistence=persistence_states['EQUIPMENT'].persistence_score,
            
            infrastructure_present=frame_cues.cue_groups['INFRASTRUCTURE']['present'],
            infrastructure_count=frame_cues.cue_groups['INFRASTRUCTURE']['count'],
            infrastructure_persistence=persistence_states['INFRASTRUCTURE'].persistence_score,
            
            # Multi-cue decision
            multi_cue_pass=multi_cue_decision.passed,
            num_cues_sustained=multi_cue_decision.num_sustained_cues,
            sustained_cues=','.join(multi_cue_decision.sustained_cues),
            
            # State
            state=current_state.value,
            state_confidence=self.state_machine.get_state_confidence()
        )
        
        self.frame_results.append(result)
        return result
    
    def process_video(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        stride: int = 1
    ) -> List[FrameResult]:
        """
        Process entire video through pipeline.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None = all)
            stride: Process every Nth frame
            
        Returns:
            List of FrameResult
        """
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"\n🎬 Processing video: {Path(video_path).name}")
        print(f"   FPS: {fps:.1f}, Total frames: {total_frames}, Stride: {stride}")
        
        self.frame_results.clear()
        self.reset()
        
        frame_id = 0
        with tqdm(total=total_frames//stride, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_id % stride == 0:
                    timestamp = frame_id / fps
                    result = self.process_frame(frame, frame_id, timestamp)
                    pbar.update(1)
                
                frame_id += 1
                if max_frames and frame_id >= max_frames:
                    break
        
        cap.release()
        
        print(f"\n✓ Processed {len(self.frame_results)} frames")
        self._print_summary()
        
        return self.frame_results
    
    def save_results(self, output_path: str):
        """Save results to CSV."""
        if not self.frame_results:
            print("⚠️  No results to save")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.frame_results])
        df.to_csv(output_path, index=False)
        print(f"✓ Saved results to: {output_path}")
    
    def reset(self):
        """Reset pipeline to initial state."""
        self.persistence_tracker.reset()
        self.state_machine.reset()
        self.frame_results.clear()
    
    def _print_summary(self):
        """Print pipeline processing summary."""
        if not self.frame_results:
            return
        
        # Count state occurrences
        state_counts = {}
        for result in self.frame_results:
            state_counts[result.state] = state_counts.get(result.state, 0) + 1
        
        # Count transitions
        transitions = self.state_machine.get_transition_summary()
        
        print("\n📊 Pipeline Summary:")
        print(f"   States distribution:")
        for state, count in state_counts.items():
            pct = 100 * count / len(self.frame_results)
            print(f"     {state:12} {count:5} frames ({pct:5.1f}%)")
        
        print(f"\n   State transitions: {transitions['total_transitions']}")
        for t in transitions['transitions']:
            print(f"     Frame {t['frame']:5} @ {t['time']:8} - {t['transition']}")


if __name__ == "__main__":
    print("Multi-Cue Pipeline initialized")
    print("Ready for video processing")
