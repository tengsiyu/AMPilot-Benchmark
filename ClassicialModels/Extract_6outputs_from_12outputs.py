import json
import os
from typing import Any, Dict, List


def _trim_output_trajectory(entry: Dict[str, Any], keep: int = 6) -> bool:
    """Trim entry['output']['Trajectory'] to the first `keep` points if present.

    Returns True if a trajectory was found and trimmed, otherwise False.
    """
    try:
        traj = entry["output"]["Trajectory"]
        if isinstance(traj, list) and len(traj) > keep:
            entry["output"]["Trajectory"] = traj[:keep]
            return True
    except Exception:
        pass
    return False


def extract_first6_from_12(
    # input_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_predicted_proportional_1000_251027.json",
    # output_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_predicted_proportional_1000_251027_6outputs.json",
    # input_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_predicted_proportional_1000_251027.json",
    # output_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_predicted_proportional_1000_251027_6outputs.json",
    # input_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_test_proportional_1000_251027.json",
    # output_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/0_5_test_proportional_1000_251027.json",
    # input_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/5_10_test_proportional_1000_251027.json",
    # output_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/5_10_test_proportional_1000_251027_6outputs.json",
    input_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/10_25_test_proportional_1000_251027.json",
    output_file: str = "/file_system/vepfs/algorithm/siyu.teng/MarineData/10_25_test_proportional_1000_251027_6outputs.json",
    keep: int = 6,
) -> None:
    """Load a large JSON array and trim each output->Trajectory from 12 to 6 points.

    - input_file: path to the original JSON containing 12-point trajectories.
    - output_file: path to save the processed JSON with 6-point trajectories.
    - keep: number of points to keep from the start of each trajectory.
    """
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    print(f"Loading: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    if not isinstance(data, list):
        print("Unexpected JSON structure: expected a top-level list.")
        return

    total = len(data)
    changed = 0
    missing = 0
    for i, entry in enumerate(data):
        ok = _trim_output_trajectory(entry, keep=keep)
        if ok:
            changed += 1
        else:
            missing += 1
        if (i + 1) % 50000 == 0:
            print(f"Processed {i + 1}/{total}...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    print(f"Done. Entries: {total}, trimmed: {changed}, without-traj-or-short: {missing}.\nSaved to: {output_file}")


if __name__ == "__main__":
    # Default execution using the requested source file and saving alongside it
    extract_first6_from_12()
