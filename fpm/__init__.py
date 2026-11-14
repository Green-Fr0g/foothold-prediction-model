import gymnasium as gym

for task_id, env_cfg, agent_cfg in (
    ("FPM-G1-v0", "FpmEnvCfg", "FpmPPORunnerCfg"),
    ("FPM-G1-Play-v0", "FpmEnvCfg_PLAY", "FpmPPORunnerCfg"),
    ("FPM-G1-Validate-Locomotion-v0", "FpmLocomotionValidationEnvCfg", "FpmPPORunnerCfg"),
    ("FPM-G1-Validate-FootPred-v0", "FpmFootPredValidationEnvCfg", "FpmPPORunnerCfg"),
    ("FPM-G1-Validate-Corridor-v0", "FpmCorridorValidationEnvCfg", "FpmPPORunnerCfg"),
    ("FPM-G1-23-v0", "FpmG1_23EnvCfg", "FpmG1_23PPORunnerCfg"),
    ("FPM-G1-23-Play-v0", "FpmG1_23EnvCfg_PLAY", "FpmG1_23PPORunnerCfg"),
    ("FPM-G1-23-Validate-Locomotion-v0", "FpmG1_23LocomotionValidationEnvCfg", "FpmG1_23PPORunnerCfg"),
    ("FPM-G1-23-Validate-FootPred-v0", "FpmG1_23FootPredValidationEnvCfg", "FpmG1_23PPORunnerCfg"),
    ("FPM-G1-23-Validate-Corridor-v0", "FpmG1_23CorridorValidationEnvCfg", "FpmG1_23PPORunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:{agent_cfg}",
        },
    )
