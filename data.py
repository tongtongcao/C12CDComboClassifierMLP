import numpy as np


def read_data(filename):
    """
    Read CVT binary-classification training data.

    Each line contains 40 values:

    SVT crosses for regions from 1 to 3:
        x, y, z, region, mask

    BMT clusters for layers from 1 to 6:
        BMT-C clusters:
            r, z, layer+3, mask
        BMT-Z clusters:
            r, phi, layer+3, mask

    label:
        1 = real
        0 = fake

    Total:
        5 * 3 + 4 * 6 + 1 = 40 values

    Returns
    -------
    features : np.ndarray
        shape [N, 39]

        For each layer:
            [xo, yo, zo, xe, ye, ze, mask]

    labels : np.ndarray
        shape [N, 1]
    """

    features = []
    labels = []

    with open(filename, "r") as f:
        lines = [
            line.strip()
            for line in f
            if line.strip() != ""
        ]

    for i, line in enumerate(lines):

        vals = [float(x) for x in line.split(",")]

        if len(vals) != 40:
            raise ValueError(
                f"Line {i+1}: expected 40 values "
                f"(39 input + 1 label), "
                f"got {len(vals)}"
            )

        # --------------------------------------------------
        # First 39 values as input features
        # --------------------------------------------------
        input_values = vals[:39]

        # --------------------------------------------------
        # Last value: binary label
        # --------------------------------------------------
        state_values = vals[39:40]

        features.append(input_values)
        labels.append(state_values)

    features = np.array(
        features,
        dtype=np.float32
    )

    targets = np.array(
        labels,
        dtype=np.float32
    )

    return features, labels


# Example usage
if __name__ == "__main__":

    features, labels = read_data(
        "realFakeCrossCombos.csv"
    )

    print(f"Read {len(features)} samples")

    print("\nInput shape :", features.shape)
    print("Label shape:", labels.shape)

    print("\nFirst sample:")

    print("input  =", features[0])
    print("label =", labels[0])