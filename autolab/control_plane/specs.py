"""Anthropic-format tool schemas for all 16 control-plane tools.

14 existing tools + sleep + finalize (agent-only, intercepted by AgentLoop).
Imported by ControlPlane.tool_specs() and used as the ``tools`` argument to
the Anthropic messages.create() call.
"""

from __future__ import annotations

TOOL_SPECS: list[dict] = [
    # ── Write tools ──────────────────────────────────────────────────────────

    {
        "name": "start_run",
        "description": (
            "Start a new training run with the given config. Returns immediately with "
            "run_id and config_id. The run begins training in the background. "
            "Only one run can be actively training at a time; if another run is active "
            "it will be paused first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": (
                        "Full training config dict. Required top-level keys: "
                        "model_name, dataset_name, grpo (dict of TRL GRPOConfig fields). "
                        "Optional: dataset_max_level, system_prompt, eval."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why this run is being started. Stored in the run record.",
                },
            },
            "required": ["config", "reason"],
        },
    },

    {
        "name": "fork",
        "description": (
            "Fork an existing run into a new child run with config overrides. "
            "Copies the parent's latest checkpoint so the child continues from where the "
            "parent left off. The parent is paused; the child starts training immediately. "
            "Use fork when you want to branch exploration (different LR, beta, etc.) "
            "without losing the parent's progress."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "parent_run_id": {
                    "type": "string",
                    "description": "run_id of the run to fork. Must be running or paused.",
                },
                "overrides": {
                    "type": "object",
                    "description": (
                        "Nested config overrides applied on top of the parent's config. "
                        "Example: {\"grpo\": {\"learning_rate\": 1e-6}}. "
                        "Any config key can be overridden (fork is not limited to the hot-modify allowlist)."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why this fork is being created.",
                },
            },
            "required": ["parent_run_id", "overrides", "reason"],
        },
    },

    {
        "name": "kill",
        "description": (
            "Permanently stop a run. The run cannot be resumed after being killed. "
            "Idempotent — killing an already-killed run is fine. "
            "If you want to stop a run temporarily and resume later, use set_active(None) instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to kill."},
                "reason": {"type": "string", "description": "Why this run is being killed."},
            },
            "required": ["run_id", "reason"],
        },
    },

    {
        "name": "modify",
        "description": (
            "Hot-modify a parameter on a running or paused run without forking. "
            "Only these parameters can be changed: learning_rate, beta, epsilon, "
            "temperature, top_p, top_k. "
            "The change is applied at the next training step. "
            "For any other parameter, use fork() with the desired override."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to modify."},
                "overrides": {
                    "type": "object",
                    "description": (
                        "Flat dict of parameter changes. "
                        "Example: {\"learning_rate\": 2e-6, \"beta\": 0.01}. "
                        "All keys must be in the hot-modify allowlist."
                    ),
                },
                "reason": {"type": "string", "description": "Why this modification is being made."},
            },
            "required": ["run_id", "overrides", "reason"],
        },
    },

    {
        "name": "set_active",
        "description": (
            "Set which run is actively training on the GPU. "
            "Pass run_id to resume a paused run or start a new active run. "
            "Pass null to pause the currently active run (GPU goes idle). "
            "Idempotent — calling with the already-active run_id is safe."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {
                    "type": ["string", "null"],
                    "description": "run_id to make active, or null to pause the active run.",
                },
                "reason": {"type": "string", "description": "Why the active run is changing."},
            },
            "required": ["run_id", "reason"],
        },
    },

    {
        "name": "set_cadence",
        "description": (
            "Adjust how often telemetry observations and samples are captured for a run. "
            "observation_cadence: emit an ObservationEvent every N steps (bounds: [10, 200]). "
            "sample_cadence: capture one (prompt, completion, reward) sample per N steps (bounds: [5, 100]). "
            "Values outside bounds are clamped silently. "
            "Use a lower cadence (more frequent) when you need to observe early dynamics; "
            "use a higher cadence (less frequent) when a run is training smoothly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to adjust cadence for."},
                "observation_cadence": {
                    "type": "integer",
                    "description": "Emit ObservationEvent every N steps. Clamped to [10, 200].",
                },
                "sample_cadence": {
                    "type": "integer",
                    "description": "Capture one sample per N steps. Clamped to [5, 100].",
                },
                "reason": {"type": "string", "description": "Why the cadence is changing."},
            },
            "required": ["run_id", "observation_cadence", "sample_cadence", "reason"],
        },
    },

    {
        "name": "eval",
        "description": (
            "Run an evaluation benchmark on a run's latest checkpoint. "
            "Returns immediately with eval_id and status='running'. Training is paused for eval. "
            "Use get_eval(eval_id) to poll for results, or sleep() — you'll be woken when it completes. "
            "After eval finishes, the system goes idle; call set_active(run_id) to resume training."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to evaluate."},
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark name. Currently supported: 'math'.",
                    "enum": ["math"],
                },
                "n_samples": {
                    "type": "integer",
                    "description": "Number of eval samples to use. -1 for full test split.",
                },
                "reason": {"type": "string", "description": "Why this eval is being run."},
            },
            "required": ["run_id", "benchmark", "n_samples", "reason"],
        },
    },

    # ── Read tools ────────────────────────────────────────────────────────────

    {
        "name": "get_run_details",
        "description": (
            "Get a full snapshot of a run: config, current status, latest observation, "
            "all modifications, all evals, and post-hoc stats. "
            "Use this when you need deep detail on a specific run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to inspect."},
            },
            "required": ["run_id"],
        },
    },

    {
        "name": "list_runs",
        "description": (
            "List all known runs with their current status, step, latest accuracy, "
            "and creation reason. Optionally filter by status. "
            "Use this to get a quick overview of all runs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status_filter": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of statuses to include. "
                        "Valid values: 'running', 'paused', 'done', 'failed', 'killed'. "
                        "Null returns all runs."
                    ),
                },
            },
            "required": [],
        },
    },

    {
        "name": "get_history",
        "description": (
            "Get the time-series of ObservationEvents for a run. "
            "Each event contains metrics (loss, reward, accuracy), trends (slope, mean, std), "
            "and any anomalies detected at that step. "
            "Optionally filter to a step range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to get history for."},
                "step_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Optional [min_step, max_step] inclusive. Null returns all steps.",
                },
                "fields": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "Optional list of metric fields to include. Null returns all.",
                },
            },
            "required": ["run_id"],
        },
    },

    {
        "name": "get_sample",
        "description": (
            "Get captured (prompt, completion, reward) samples from a run. "
            "Useful for understanding what the model is generating and whether "
            "it's producing valid answer formats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to get samples from."},
                "step": {
                    "type": ["integer", "null"],
                    "description": "Optional step to filter samples near. Null returns most recent.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of samples to return. Default 5.",
                    "default": 5,
                },
            },
            "required": ["run_id"],
        },
    },

    {
        "name": "compute_trend",
        "description": (
            "Compute the trend (slope, mean, std) of a metric over a recent window of steps. "
            "Useful for answering 'is loss still decreasing?' or 'has reward plateaued?'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "run_id to analyze."},
                "metric": {
                    "type": "string",
                    "description": (
                        "Metric name to trend. Examples: 'loss', "
                        "'rewards/accuracy_reward/mean', 'reward'."
                    ),
                },
                "window": {
                    "type": "integer",
                    "description": "Number of most recent observations to use. Default 20.",
                    "default": 20,
                },
            },
            "required": ["run_id", "metric"],
        },
    },

    {
        "name": "get_config",
        "description": "Get the full config dict for a config_id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config_id": {"type": "string", "description": "Config ID (SHA-256[:16])."},
            },
            "required": ["config_id"],
        },
    },

    {
        "name": "get_eval",
        "description": (
            "Get the result of an evaluation by eval_id. "
            "Returns status ('running', 'done', 'failed') and, when done, "
            "accuracy, by_level, and by_type breakdowns."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "eval_id": {"type": "string", "description": "eval_id returned by eval()."},
            },
            "required": ["eval_id"],
        },
    },

    # ── Agent-only tools (intercepted by AgentLoop, not routed to ControlPlane) ──

    {
        "name": "sleep",
        "description": (
            "Pause the agent loop for up to `seconds` seconds. "
            "The loop can be woken early by: a hard anomaly (NaN loss, OOM), "
            "a pending evaluation completing, a human message arriving, or the .stop interrupt. "
            "The response includes `woken_by` ('timer' | 'anomaly' | 'eval_complete' | 'human' | 'interrupted') "
            "and `actual_seconds` elapsed. "
            "This is the only way to yield between turns. Always end a turn with sleep() unless "
            "you are about to call finalize()."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "number",
                    "description": "Maximum seconds to sleep. Typical values: 60-600.",
                },
                "reason": {
                    "type": "string",
                    "description": "Why you are sleeping (what you are waiting for).",
                },
            },
            "required": ["seconds", "reason"],
        },
    },

    {
        "name": "finalize",
        "description": (
            "Declare the session complete. Writes the winning run and summary to the session record. "
            "No further tool calls are accepted after this. "
            "Call this when you have met the success criterion, exhausted the budget, "
            "or been asked to stop by a human."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "winning_run_id": {
                    "type": ["string", "null"],
                    "description": "run_id of the best run found. Null if no run was trained.",
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "Brief explanation of what was tried, what worked, and why this run won. "
                        "Visible to the human researcher reviewing the session."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "Why the session is being finalized now.",
                },
            },
            "required": ["winning_run_id", "summary", "reason"],
        },
    },
]
