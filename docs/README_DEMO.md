# Demo Mode

Use the `--demo` flag on `stereo_vision_app` to run a deterministic demo using sample data in `data/stereo_images`.

Output: `reports/demo_output/` will contain `disparity.png`, `cloud.ply`, and `run.log`.

If models are present in `models/` the app will try neural matchers; otherwise it will fall back to CPU block matching.
