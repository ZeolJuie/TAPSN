import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(path):
    with Path(path).open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description='Plot ego sensor gait motion from sensor_pose.csv')
    parser.add_argument('sensor_pose_csv')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    rows = load_rows(args.sensor_pose_csv)
    if not rows:
        raise RuntimeError('sensor_pose.csv is empty')

    frames = [int(row['frame']) for row in rows]
    unique_frames = sorted(set(frames))
    frame_to_t = {frame: idx for idx, frame in enumerate(unique_frames)}

    speed_by_frame = {}
    waist_z = []
    head_z = []
    head_pitch = []

    for row in rows:
        frame = int(row['frame'])
        t = frame_to_t[frame]
        speed_by_frame[frame] = float(row['ego_speed'])
        mount = row['mount']
        gait_z = float(row['gait_z_offset'])
        gait_pitch = float(row['gait_pitch_offset_deg'])
        if mount == 'waist' and row['sensor_name'] == 'CAM_FRONT':
            waist_z.append((t, gait_z))
        if mount == 'head' and row['sensor_name'] == 'CAM_TOP':
            head_z.append((t, gait_z))
            head_pitch.append((t, gait_pitch))

    ts = list(range(len(unique_frames)))
    speed = [speed_by_frame[frame] for frame in unique_frames]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(ts, speed, label='ego speed (m/s)', color='black')
    axes[0].set_ylabel('speed')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if waist_z:
        axes[1].plot([t for t, _ in waist_z], [z for _, z in waist_z], label='waist z offset (m)', color='tab:blue')
    if head_z:
        axes[1].plot([t for t, _ in head_z], [z for _, z in head_z], label='head z offset (m)', color='tab:orange')
    axes[1].set_ylabel('z offset')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    if head_pitch:
        axes[2].plot(
            [t for t, _ in head_pitch],
            [p for _, p in head_pitch],
            label='head pitch offset (deg)',
            color='tab:red',
        )
    axes[2].set_ylabel('pitch')
    axes[2].set_xlabel('sample index')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()
    output = args.output or str(Path(args.sensor_pose_csv).with_name('sensor_gait_motion.png'))
    fig.savefig(output, dpi=150)
    print(output)


if __name__ == '__main__':
    main()
