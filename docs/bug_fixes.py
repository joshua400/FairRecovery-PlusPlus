"""
FAIRRECOVERY++ — CRITICAL BUG FIXES
Apply these patches before submission.
"""

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: fairrecovery_environment.py
# Bug: pending_allocations cleared before reward engine checks them,
#      causing persistent_ignore_vulnerable to fire on EVERY execute step.
# ─────────────────────────────────────────────────────────────────────────────

# Replace the "execute" branch in step() with this:

        elif action_type == "execute":
            city.snapshot_services()

            # ⚠️ CAPTURE allocated zones BEFORE apply_allocations() clears them
            allocated_zone_ids_snapshot = frozenset(
                int(a.get("zone", -1))
                for a in city.pending_allocations
                if a.get("zone") is not None
            )

            exec_violations = city.apply_allocations()
            all_violations  = violations + exec_violations

            components = self._reward_engine.compute_execute_step(
                city=city,
                violations=all_violations,
                allocated_zone_ids=allocated_zone_ids_snapshot,   # ← NEW param
            )
            reward   = components.R_total
            r_exec   = components.R_exec
            r_fair   = components.R_fair
            r_safe   = components.R_safe
            feedback = components.feedback
            city.record(f"executed | {feedback}")
            city.step_stage = "analyze"

            if city.day >= MAX_DAYS or city.budget_left <= 0:
                done = True


# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: rewards.py — RewardEngine.compute_execute_step
# Accept allocated_zone_ids as a parameter instead of reading from city.
# ─────────────────────────────────────────────────────────────────────────────

    def compute_execute_step(
        self,
        city: CityState,
        violations: List[str],
        allocated_zone_ids: frozenset = frozenset(),   # ← NEW param with default
    ) -> RewardComponents:

        self._step_count += 1

        # Use passed-in snapshot, not city.pending_allocations (already cleared)
        vuln_zone_ids = {z.zone_id for z in city.zones if z.is_vulnerable}

        if vuln_zone_ids:
            history_text = " ".join(city.history)
            zone_served_in_history = any(
                f"zone {zid}" in history_text.lower() or str(zid) in history_text
                for zid in vuln_zone_ids
            )
            # Also check current step's allocations via snapshot
            zone_served_this_step = bool(vuln_zone_ids & allocated_zone_ids)
            zone_served = zone_served_in_history or zone_served_this_step

            if not zone_served and city.day > 1:
                self._vulnerable_ignored_days += 1
                if self._vulnerable_ignored_days >= 2:
                    violations.append(f"persistent_ignore_vulnerable:{vuln_zone_ids}")
            else:
                # Reset counter when served
                self._vulnerable_ignored_days = max(0, self._vulnerable_ignored_days - 1)

        R_exec = compute_exec_reward(city.prev_services, city.zones)
        R_fair = compute_fairness_reward(city.zones)
        R_safe = compute_safety_reward(violations)

        w = REWARD_WEIGHTS
        R_total = (
            w["exec"] * R_exec +
            w["fair"] * R_fair +
            w["safe"] * R_safe
        )
        R_total = float(max(-1.0, min(1.0, R_total)))
        self._cumulative_reward += R_total

        # ... rest of method unchanged


# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: fairrecovery_environment.py — rubric score must flow to obs.reward
# Bug: rubric scores added to cumulative_reward but not step reward,
#      so GRPO training never sees terminal fairness/utility bonuses.
# ─────────────────────────────────────────────────────────────────────────────

        # Replace this block at the end of step():
        rubric_score = self._rubrics.forward(typed_action, obs)
        if rubric_score != 0.0:
            obs.cumulative_reward += rubric_score
            obs.reward = round(obs.reward + rubric_score, 4)   # ← ADD THIS LINE


# ─────────────────────────────────────────────────────────────────────────────
# FIX 4: fairrecovery_environment.py — state as method not property
# Bug: OpenEnv's FastAPI wrapper calls env.state() but it's a @property.
# ─────────────────────────────────────────────────────────────────────────────

    # Remove @property decorator — make it a regular method:
    def state(self) -> FairRecoveryState:   # NOT @property
        """Internal state exposed via GET /state."""
        if self._city is None:
            return FairRecoveryState()
        # ... rest unchanged


# ─────────────────────────────────────────────────────────────────────────────
# FIX 5: state.py — zone_id type coercion
# Bug: zone_id from JSON can arrive as string, breaking integer comparison.
# ─────────────────────────────────────────────────────────────────────────────

    def apply_allocations(self) -> List[str]:
        violations: List[str] = []

        for alloc in self.pending_allocations:
            raw_zone_id = alloc.get("zone")
            resource    = alloc.get("resource")

            # ← ADD THIS: coerce to int safely
            try:
                zone_id = int(raw_zone_id)
            except (TypeError, ValueError):
                violations.append(f"invalid_zone:{raw_zone_id}")
                self.violations_total += 1
                continue

            if not (0 <= zone_id < len(self.zones)):
                violations.append(f"invalid_zone:{zone_id}")
                self.violations_total += 1
                continue

            # ... rest unchanged


# ─────────────────────────────────────────────────────────────────────────────
# FIX 6: models.py — remove duplicate class definitions
# Bug: AllocationItem and ZoneObservation defined twice; second shadows first.
# ─────────────────────────────────────────────────────────────────────────────

# Delete lines 104–119 in models.py (the first AllocationItem and ZoneObservation
# that inherit from BaseAction/BaseObservation). Keep only the _BaseModel versions.


# ─────────────────────────────────────────────────────────────────────────────
# FIX 7: models.py — add explicit done/reward fields to FairRecoveryObservation
# Safety net: if OpenEnv BaseObservation fallback (plain BaseModel) is used,
# these fields won't exist and _build_observation() will throw.
# ─────────────────────────────────────────────────────────────────────────────

# Add to FairRecoveryObservation:
    reward: float = Field(
        default=0.0,
        description="Step reward received from the last action.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    agent_events: List[str] = Field(
        default_factory=list,
        description="Events emitted by dynamic agents this step.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIX 8: inference.py — fix undefined service_level attribute
# Bug: HFInferencePolicy._build_prompt references z.service_level
#      but ZoneObservation has z.service
# ─────────────────────────────────────────────────────────────────────────────

    def _build_prompt(self, obs: FairRecoveryObservation) -> str:
        zones_info = "\n".join([
            f"Zone {z.zone_id}: Damage={z.damage:.2f}, "
            f"Vulnerability={z.vulnerable_ratio:.2f}, "
            f"Svc={z.service:.2f}"          # ← was z.service_level, fix to z.service
            for z in obs.zones
        ])
        # ... rest unchanged


# ─────────────────────────────────────────────────────────────────────────────
# REWARD TUNING (optional but recommended for better training delta)
# ─────────────────────────────────────────────────────────────────────────────

# In constants.py, change:
REWARD_WEIGHTS = {
    "exec": 0.5,   # ← was 1.0; reduce to stop exec dominating
    "fair": 1.0,   # ← was 0.5; DOUBLE fairness weight
    "safe": 0.5,   # unchanged
}

# This makes the hard scenario's fairness trap actually matter in training.
# Agents ignoring Zone 4 will now score noticeably lower than fairness-aware ones.

# Also in rewards.py::compute_analysis_step, change:
R_total = 0.3 * R_analysis  # ← was 0.1; increase so analysis quality matters
