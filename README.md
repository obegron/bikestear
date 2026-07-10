# ftms2pad

`ftms2pad` turns one FTMS indoor bike and one webcam into a Linux virtual gamepad.

- X follows horizontal torso position relative to a neutral calibration.
- Y follows `speed_kph`, `watts`, or `cadence_rpm` from the bike.
- The controller output loop runs at 60 Hz by default.

Vision uses only MediaPipe shoulder and hip pose landmarks. It does not detect or track the head or face.

## Requirements

- Linux with Bluetooth Low Energy support
- Python 3.10 through 3.12
- One V4L2 webcam
- An FTMS indoor bike, or the built-in `sim` bike for development
- `/dev/uinput` access for the `run` command

Install the Python environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The MediaPipe dependency is pinned to the final release line that provides its lightweight Pose tracking API. On the first `calibrate`, `monitor`, or `run`, MediaPipe downloads its official lite pose model (about 3 MB) into the virtual environment, so allow network access once during setup. Later runs use the local model. The default vision configuration is deliberately modest at 320x180 and 20 FPS.

## uinput permissions

Load the kernel module and add your user to the group that owns `/dev/uinput` (commonly `input`):

```bash
sudo modprobe uinput
sudo usermod -aG input "$USER"
```

Log out and back in after changing groups. On systems that do not assign `/dev/uinput` to the `input` group, install this udev rule as `/etc/udev/rules.d/70-uinput.rules`:

```udev
KERNEL=="uinput", GROUP="input", MODE="0660", OPTIONS+="static_node=uinput"
```

Then reload rules with `sudo udevadm control --reload-rules && sudo udevadm trigger`.

## Profile format

Profiles live in `profiles/<name>.yaml`. All mapping values are validated on load.

```yaml
name: supertuxkart

vision:
  camera: 0
  width: 320
  height: 180
  fps: 20
  min_confidence: 0.5

x_axis:
  gain: 1.4
  deadzone: 0.06
  smoothing: 0.25
  invert: false
  stale_after_ms: 250

y_axis:
  source: speed_kph       # speed_kph, watts, or cadence_rpm
  min: 0                  # source units
  max: 40                 # source units
  deadzone: 1             # source units above min
  smoothing: 0.15
  curve: linear
  invert: true

uinput:
  x_axis: ABS_X
  y_axis: ABS_Y
```

X is a signed value from -1 to 1. Y is normalized and clamped from 0 to 1 before it is emitted across the configured signed Linux axis. `invert` reverses the corresponding mapped direction.

## Cameras and bikes

List available V4L2 cameras and nearby BLE devices:

```bash
uv run ftms2pad list-cameras
uv run ftms2pad list-bikes
```

Select exactly one camera in the profile or override it with `--camera 2`. There is no automatic or multi-camera selection.

## Calibration

Position the camera so both shoulders and preferably both hips remain visible while riding. Run:

```bash
uv run ftms2pad calibrate --profile supertuxkart
```

Calibration collects neutral, left, and right torso positions. Each phase must contain enough reliable samples and each side must be meaningfully separated from neutral. A failed phase is not synthesized; the error identifies what to redo. Calibration is stored next to the profile as `profiles/supertuxkart.calibration.json`.

Use `--no-preview` for text-only calibration, or `--no-mirror` for an unmirrored preview.

## Monitor

Inspect the complete signal path before opening a game:

```bash
uv run ftms2pad monitor --profile supertuxkart --bike sim
uv run ftms2pad monitor --profile supertuxkart --bike '<BLE address or name>'
```

Monitor reports torso confidence and sample age, raw and mapped X, the selected raw and mapped FTMS Y value, actual vision FPS, and inference time. Press `q` in the preview to exit. Add `--no-preview` for terminal-only monitoring.

## Run

Start the virtual controller without a GUI:

```bash
uv run ftms2pad run --profile supertuxkart --bike '<BLE address or name>'
```

For a mapping and simulated-bike check that does not open `/dev/uinput`, use:

```bash
uv run ftms2pad run --profile supertuxkart --bike sim --dry-run
```

Press Ctrl+C to stop. The FTMS task is cancelled, the vision worker releases its camera and MediaPipe instance, and the virtual input device is closed.

## Performance model

Camera capture and MediaPipe inference run together in a dedicated worker thread. That worker overwrites one latest-result slot; frames never accumulate in a queue. MediaPipe uses its lightest pose model (`model_complexity=0`) in tracking mode on the CPU.

Vision intentionally runs slower than the 60 Hz gamepad loop. The output loop reuses the newest vision and FTMS samples without waiting for a camera frame. When vision becomes stale or loses the torso, X smoothly returns toward neutral while bike-driven Y continues to update.

## Tests

The deterministic suite does not need a camera, BLE device, or `/dev/uinput`:

```bash
make test
```
