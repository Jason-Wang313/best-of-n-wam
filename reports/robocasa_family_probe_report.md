# RoboCasa Family Probe Report

- status: `completed`
- probe date: `2026-05-31`
- interpreter: `C:\Users\wangz\external_benchmarks\.venvs\robocasa\Scripts\python.exe`
- candidate task IDs: `16`
- usable nonzero-distance task IDs: `12`

## Usable Tasks

- `robocasa/OpenDrawer`
- `robocasa/OpenCabinet`
- `robocasa/OpenMicrowave`
- `robocasa/TurnOnSinkFaucet`
- `robocasa/CloseDrawer`
- `robocasa/CloseCabinet`
- `robocasa/CloseMicrowave`
- `robocasa/TurnOffSinkFaucet`
- `robocasa/TurnOnStove`
- `robocasa/TurnOffStove`
- `robocasa/OpenOven`
- `robocasa/CloseOven`

## Excluded Tasks

- `robocasa/OpenDoor`: reset failed because `ManipulateDoor.__init__()` required a missing `fixture_id` argument under the generic Gymnasium registration.
- `robocasa/CloseDoor`: reset failed for the same missing `fixture_id` argument.
- `robocasa/OpenDishwasher`: reset worked, but the current observation exposed no non-counter target-to-EEF key and `object_distance()` was `0.0`.
- `robocasa/CloseDishwasher`: reset worked, but the current observation exposed no non-counter target-to-EEF key and `object_distance()` was `0.0`.

The promoted 12-task artifact therefore covers a verified local slice of RoboCasa open, close, sink, stove, microwave, cabinet, drawer, and oven tasks. It should not be described as full RoboCasa-wide validation.
