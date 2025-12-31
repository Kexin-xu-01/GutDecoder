import json
import pickle
import sys
import traceback
import types
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parent
PAYLOAD_FILE = JOB_DIR / "payload.pkl"
STATUS_FILE = JOB_DIR / "status.json"
OUTPUT_FILE = JOB_DIR / "output.pkl"
TRACEBACK_FILE = JOB_DIR / "traceback.log"

def main():
    exit_code = 0
    status = {"state": "UNKNOWN"}
    try:
        with open(PAYLOAD_FILE, "rb") as handle:
            payload = pickle.load(handle)

        # Load imports and variables
        sys.path = payload["sys_path"]
        namespace = {}
        namespace.update(payload["variables"])
        for alias, module_name in payload["modules"].items():
            try:
                module = __import__(module_name)
            except Exception:
                continue
            namespace[alias] = module

        # Execute
        exec(payload["cell"], namespace)

        # Extract output variables
        if payload.get("capture_all_outputs", False):
            vars_to_capture = []
            for name, value in namespace.items():
                if name == "__builtins__":
                    continue
                if isinstance(value, types.ModuleType):
                    continue
                vars_to_capture.append(name)
        else:
            vars_to_capture = payload["outputs"]
            for name in vars_to_capture:
                if name not in namespace:
                    raise RuntimeError("Result variable '{var}' was not defined by the job.".format(var=name))
        namespace_payload = {}
        namespace_errors = {}
        for name in vars_to_capture:
            value = namespace[name]
            try:
                namespace_payload[name] = pickle.dumps(
                    value, protocol=payload["pickle_protocol"]
                )
            except Exception as exc:
                namespace_errors[name] = repr(exc)

        # Write to pickle file
        with open(OUTPUT_FILE, "wb") as handle:
            pickle.dump(
                {
                    "namespace": namespace_payload,
                    "errors": namespace_errors,
                },
                handle,
                protocol=payload["pickle_protocol"],
                )
        status = {"state": "COMPLETED"}
    except Exception as exc:
        exit_code = 1
        status = {"state": "FAILED", "message": str(exc)}
        traceback.print_exc()
        with open(TRACEBACK_FILE, "w") as trace_handle:
            traceback.print_exc(file=trace_handle)
    finally:
        with open(STATUS_FILE, "w") as status_handle:
            json.dump(status, status_handle)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
