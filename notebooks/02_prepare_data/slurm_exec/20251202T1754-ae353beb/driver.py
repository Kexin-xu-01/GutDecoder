import json
import pickle
import sys
import traceback
import types
import importlib.util
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parent
PAYLOAD_FILE = JOB_DIR / "payload.pkl"
STATUS_FILE = JOB_DIR / "status.json"
OUTPUT_FILE = JOB_DIR / "output.pkl"
TRACEBACK_FILE = JOB_DIR / "traceback.log"
CELL_FILE = JOB_DIR / "cell.py"

HELPER_FILE = JOB_DIR / "ipy_slurm_exec_runtime.py"
_spec = importlib.util.spec_from_file_location("ipy_slurm_exec_runtime", HELPER_FILE)
_runtime = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runtime)

def main():
    exit_code = 0
    status = {"state": "UNKNOWN"}
    try:
        with open(PAYLOAD_FILE, "rb") as handle:
            payload = pickle.load(handle)

        # Load imports and variables
        sys.path = payload["sys_path"]
        capture_all_inputs = payload.get("capture_all_inputs", False)
        namespace = {}
        namespace_errors = {}
        # for name, record in payload["variables"].items():
        #     try:
        #         namespace[name] = _runtime.restore_from_record(record, JOB_DIR)
        #     except Exception as exc:
        #         mode = record.get("mode", "unknown")
        #         path = record.get("path")
        #         extra = f", path={path}" if path else ""
        #         trace_text = traceback.format_exc()
        #         namespace_errors[name] = f"{repr(exc)} [mode={mode}{extra}]\n{trace_text}"
        # if namespace_errors:
        #     detail = "\n".join(f"{var}: {err}" for var, err in sorted(namespace_errors.items()))
        #     raise RuntimeError(f"Failed to restore input variables:\n{detail}")
        # Was the above engineering really necessary?
        for name, record in payload["variables"].items():
            try:
                namespace[name] = _runtime.restore_from_record(record, JOB_DIR)
            except Exception as exc:
                if capture_all_inputs:
                    # treat as soft error
                    namespace_errors[name] = repr(exc)
                else:
                    raise
        for alias, module_name in payload["modules"].items():
            try:
                module = __import__(module_name)
            except Exception:
                continue
            namespace[alias] = module

        # Important to print import errors here, 
        # because they could be reason why cell execution fails next.
        if namespace_errors:
            print("Import errors in Slurm job:")
            for name in sorted(namespace_errors.keys()):
                print("  {var}: '{err}'".format(var=name, err=namespace_errors[name]))
        namespace_errors = {}

        # Execute
        cell_source = payload["cell"]
        try:
            CELL_FILE.write_text(cell_source)
        except Exception:
            pass
        code_obj = compile(cell_source, str(CELL_FILE), "exec")
        exec(code_obj, namespace)

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
        capture_all_outputs = payload.get("capture_all_outputs", False)
        namespace_payload = {}
        namespace_errors = {}
        for name in vars_to_capture:
            value = namespace[name]
            try:
                namespace_payload[name] = _runtime.serialize_variable(
                    name,
                    value,
                    root_dir=JOB_DIR,
                    rel_root="outputs",
                    protocol=payload["pickle_protocol"],
                )
            except Exception as exc:
                if capture_all_outputs:
                    # treat as soft-error
                    namespace_errors[name] = repr(exc)
                else:
                    raise

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
        trimmed_tb = exc.__traceback__
        driver_path = str(Path(__file__).resolve())
        # Only drop the driver frame if the next frame refers to the cell code.
        if trimmed_tb and trimmed_tb.tb_frame and str(trimmed_tb.tb_frame.f_code.co_filename) == driver_path:
            next_tb = trimmed_tb.tb_next
            if next_tb and str(next_tb.tb_frame.f_code.co_filename) == str(CELL_FILE):
                trimmed_tb = next_tb
        traceback.print_exception(type(exc), exc, trimmed_tb)
        with open(TRACEBACK_FILE, "w") as trace_handle:
            traceback.print_exception(type(exc), exc, trimmed_tb, file=trace_handle)
    finally:
        with open(STATUS_FILE, "w") as status_handle:
            json.dump(status, status_handle)
    return exit_code

if __name__ == "__main__":
    raise SystemExit(main())
