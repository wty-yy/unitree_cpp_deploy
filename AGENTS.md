# AGENTS.md

## AI Experiment and Data Directory Standards

- All paths are relative to the project root.
- Use the current local time for timestamps in `YYYYMMDD_HHMMSS` format, for example `20261025_150710`.
- Store AI-run experiments in `debug_ai/<timestamp>_<experiment_name>/`. Use a short name describing the purpose, and reuse the same directory for follow-up work on that experiment.
- Within an experiment directory, store code in `src/`, images in `fig/`, videos in `vid/`, and data in `data/`. Create subdirectories as needed.
- If the cumulative size of an experiment or a collected data batch exceeds `100 MB`, move that batch to `debug_data/<timestamp>_<data_name>/`. Apply the threshold to the entire batch, not just individual files.
- Use a short name describing the data, and reuse the same directory for follow-up work on that batch. Save data directly there when it is known in advance that the threshold will be exceeded.
- When moving bulk data out of an experiment's `data/` directory, keep code, images, and videos in the original experiment directory. Avoid retaining duplicate data copies at the original location.
- After moving data, update affected script paths or configurations and verify that the experiment can still find its data.
- Add `/debug_ai/` and `/debug_data/` to the project root `.gitignore` to ignore both directories and their contents. Preserve existing rules, and do not add duplicates when equivalent rules already exist.
