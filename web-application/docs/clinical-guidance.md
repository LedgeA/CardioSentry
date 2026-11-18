# Clinical guidance

The application has a versioned guidance layer separate from the classifier.
It uses a stored prediction's served findings and the user's symptom, age,
and history answers. It does not diagnose disease or infer measurements from
an ECG waveform.

## Urgency tiers

| Tier | Meaning |
| --- | --- |
| Green | Normal model read with no information raising urgency |
| Yellow | Routine clinician review |
| Orange | Same-day medical assessment |
| Red | Emergency assessment |

Emergency symptoms can raise urgency regardless of model scores. The highest
applicable finding or combination determines the result. Pediatric and paced
ECGs have separate cautions and escalation rules. Missing context is shown
explicitly. The browser submits a run ID and context; the server resolves the
stored prediction and validates its descriptor and served labels.

## Rules and sources

`backend/services/guidance.py` contains the rule version, per-condition rules,
combination rules, escalation logic, and cited source records. Source URLs and
identifiers are returned with the guidance result. `tests/test_guidance.py`
checks tiers, symptom overrides, context handling, and supported label policy.

The configured classifier descriptor must match a supported rule set exactly.
