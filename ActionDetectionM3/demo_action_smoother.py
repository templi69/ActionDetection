from action_smoother import ActionSmoother


smoother = ActionSmoother(
    window_size=7,
    stable_confidence_threshold=0.15
)


test_predictions = [
    ("contact juggling", 0.40),
    ("contact juggling", 0.43),
    ("contact juggling", 0.45),
    ("stretching arm", 0.24),
    ("contact juggling", 0.42),
    ("contact juggling", 0.39),
    ("contact juggling", 0.44),
]


for action, confidence in test_predictions:

    result = smoother.update(
        action,
        confidence
    )

    print(
        f"Raw: {action:20s} "
        f"{confidence:.2f}"
        f"  →  Stable: "
        f"{result['action']:20s} "
        f"{result['confidence']:.2f}"
    )