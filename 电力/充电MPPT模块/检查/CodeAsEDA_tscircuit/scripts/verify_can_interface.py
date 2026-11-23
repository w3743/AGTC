"""Check only the isolated CAN interface preview against its source-derived reference."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
POWER = PROJECT.parents[2]
sys.path.insert(0, str(POWER / "检查/联合仿真/code_as_eda"))
from reference import write_json
from verify import verify_circuit


def main():
    directory = PROJECT / "exports/can_interface_preview"
    report = verify_circuit(PROJECT / "reference_can_interface.json", directory, 10)
    report["scope"] = ("PARTIAL INTERFACE ONLY: isolated CAN transceiver, connector and local decoupling. "
                       "The charger board and its power/MCU paths are not in this check.")
    report["board_release_status"] = "NOT_COMPLETE"
    report["external_net_boundaries"] = ["CAN_5V_SW", "MCU_3V3", "GND_SYS", "CAN_RX", "CAN_TX"]
    write_json(directory / "netlist_verify_report.json", report)
    pins = report["pin_net_comparison"]
    print(f"{report['status']} CAN interface only: {report['components']['observed']}/10 components, "
          f"{pins['equal']}/{pins['total']} pins, {len(report['findings'])} differences.")
    raise SystemExit(report["status"] != "PASS")


if __name__ == "__main__":
    main()
