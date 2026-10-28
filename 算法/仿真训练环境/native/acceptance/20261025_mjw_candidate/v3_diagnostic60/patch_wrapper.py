from pathlib import Path
p = Path('/mnt/c/pioneer/先进通用履带底盘工程/算法/仿真训练环境/native/acceptance/20261025_v3_diagnostic60/diagnostic_wrapper.py')
s = p.read_text(encoding='utf-8')
def exact(old, new):
    global s
    if old not in s:
        raise RuntimeError('missing expected text: ' + old[:100])
    s = s.replace(old, new, 1)
def region(start, end, new):
    global s
    i = s.index(start)
    j = s.index(end, i)
    s = s[:i] + new + s[j:]
exact('if str(PROTO) not in sys.path:\n    sys.path.insert(0, str(PROTO))',
      'if str(PROTO) not in sys.path:\n    sys.path.insert(0, str(PROTO))\nif str(TRAIN.parent) not in sys.path:\n    sys.path.insert(0, str(TRAIN.parent))')
exact('def _contacts_snapshot(backend: NewtonXPBDGBackend) -> dict[str, np.ndarray]:    result: dict[str, np.ndarray] = {}',
      'def _contacts_snapshot(backend: NewtonXPBDGBackend) -> dict[str, np.ndarray]:\n    result: dict[str, np.ndarray] = {}')
region('        "bad_body_rows_by_state_field": bad,', '        "backend_call_index_zero_based":',
       '''        "nonfinite_rows_by_field": bad,
        "world_ids": sorted({int(world) for key, rows in bad.items()
                              if key.startswith(("state_0.body_", "state_1.body_"))
                              for row in rows
                              for world in [int(backend.model.body_world.numpy()[row])]}),
''')
region('def _instrumented_step(self: NewtonXPBDGBackend, actions: np.ndarray) -> None:',
       'def _instrumented_read(self: NewtonXPBDGBackend) -> dict[str, np.ndarray]:',
'''def _instrumented_step(self: NewtonXPBDGBackend, actions: np.ndarray) -> None:
    call_index = int(getattr(self, "_diag_policy_call_index", 0))
    self._diag_policy_call_index = call_index + 1
    frame: dict[str, Any] = {
        "call_index_zero_based": call_index,
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pre": _snapshot(self, actions),
    }
    _HISTORY.append(frame)
    self._diag_current_frame = frame
    self._diag_in_backend_step = True
    try:
        _ORIGINAL_STEP(self, actions)
    except BaseException as exc:
        self._diag_in_backend_step = False
        frame["exception"] = f"{type(exc).__name__}: {exc}"
        frame["post"] = _snapshot(self)
        bad = _nonfinite_fields(frame["post"])
        bad.update({f"internal_read.{key}": rows for key, rows
                    in frame.get("internal_read_nonfinite_fields", {}).items()})
        _save_event(self, "backend_step_exception", bad,
                    frame.get("internal_read_values"), exception=frame["exception"])
        raise
    finally:
        self._diag_in_backend_step = False
    frame["post"] = _snapshot(self)
    bad = _nonfinite_fields(frame["post"])
    bad.update({f"internal_read.{key}": rows for key, rows
                in frame.get("internal_read_nonfinite_fields", {}).items()})
    if bad:
        _save_event(self, "nonfinite_backend_snapshot_after_policy_step", bad,
                    frame.get("internal_read_values"))


''')
region('def _instrumented_read(self: NewtonXPBDGBackend) -> dict[str, np.ndarray]:',
       'print("DIAGNOSTIC_MANIFEST "',
'''def _instrumented_read(self: NewtonXPBDGBackend) -> dict[str, np.ndarray]:
    values = _ORIGINAL_READ(self)
    read_bad = _nonfinite_fields(values)
    frame = getattr(self, "_diag_current_frame", None)
    if getattr(self, "_diag_in_backend_step", False):
        if read_bad and frame is not None:
            frame["internal_read_nonfinite_fields"] = read_bad
            frame["internal_read_values"] = {name: np.asarray(value).copy()
                                             for name, value in values.items()
                                             if isinstance(value, np.ndarray)}
    elif read_bad:
        if frame is not None:
            frame["post_read"] = {name: np.asarray(value).copy() for name, value in values.items()
                                  if isinstance(value, np.ndarray)}
        bad = _nonfinite_fields(frame.get("post", {})) if frame else {}
        bad.update({f"read.{key}": rows for key, rows in read_bad.items()})
        _save_event(self, "nonfinite_backend_read_before_env_reset", bad, values)
    return values


NewtonXPBDGBackend.step = _instrumented_step
NewtonXPBDGBackend.read = _instrumented_read
''')
p.write_text(s, encoding='utf-8', newline='\n')
print('patched', p, 'bytes', p.stat().st_size)