"""
Custom CNN classifier for marine debris detection.

Placeholder: to be implemented after U-Net.
Plan:
    Input (6 bands)
        |
    Conv(32) + ReLU -> MaxPool
    Conv(64) + ReLU -> MaxPool
    Conv(128) + ReLU
    Flatten -> Dense(128) -> Dense(2) -> Softmax
        |
    Output: patch-level classification
"""
