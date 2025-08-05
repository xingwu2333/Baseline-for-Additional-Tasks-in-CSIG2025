This is the baseline for additional tasks of “The 6th Chinese Conference on China Society of lmage and Graphics 2025” using Detection-based Tracking model, and detection results are obtained through the detection algorithm of the official toolkit of the challenge.

track.py: Track the detection results.
Relevant parameters: 
max_age=30, # Maximum unmatched lifetime 
min_hits=1, # Minimum number of hits 
threshold=11, # Correlation threshold 
max_gap=1, # Maximum number of frames between correlations

merge_tracks.py: Merge tracks on tracking results to improve track completeness and score, use on demand.

val_scores.txt: Performance score of the validation set on the competition platform.

Requirement: filterpy, scipy, collections, etc.
