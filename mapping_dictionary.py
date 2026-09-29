import re
import xml.etree.ElementTree as ET

import h5py
import numpy as np

import config


def inspect_typeid_mapping(hdf5_path=None, save_report=False, report_path="typeid_report.txt"):
    hdf5_path = hdf5_path or config.CONCAT_OUTPUT_PATH

    with h5py.File(hdf5_path, "r") as h5_file:
        acq_xml = h5_file["RawData/AcquisitionTaskDescription"][0].tobytes().decode(
            "utf-8", "ignore"
        )
        sfreq_match = re.search(r"<SamplingFrequency>(\d+)</SamplingFrequency>", acq_xml)
        sfreq = float(sfreq_match.group(1)) if sfreq_match else config.DEFAULT_SFREQ

        grp = h5_file["AsynchronData"]
        time_idx = np.array(grp["Time"]).reshape(-1) if "Time" in grp else np.array([])
        type_id = np.array(grp["TypeID"]).reshape(-1) if "TypeID" in grp else np.array([])
        value = np.array(grp["Value"]).reshape(-1) if "Value" in grp else np.array([])

        type_map = {}
        if "AsynchronSignalTypes" in grp:
            xml_text = grp["AsynchronSignalTypes"][0].tobytes().decode("utf-8", "ignore")
            root = ET.fromstring(xml_text)
            for desc in root.findall(".//AsynchronSignalDescription"):
                signal_id = desc.findtext("ID")
                signal_desc = desc.findtext("Description") or desc.findtext("Name") or ""
                if signal_id is not None:
                    type_map[int(signal_id)] = signal_desc

    print("=" * 70)
    print("TypeID mapping viewer")
    print("=" * 70)
    print(f"Sampling frequency: {sfreq} Hz")
    print(f"Event count: {len(time_idx)}")
    print(f"Unique TypeID values: {np.unique(type_id) if type_id.size else []}")

    print("\nTypeID -> label")
    for id_num in sorted(type_map):
        print(f"  {id_num:3d} -> {type_map[id_num]}")

    print("\nValid events")
    valid_count = 0
    for sample_idx, marker_type, marker_value in zip(time_idx, type_id, value):
        if marker_value != 1:
            continue
        label = type_map.get(int(marker_type), f"unknown_{int(marker_type)}")
        time_sec = float(sample_idx) / sfreq
        print(f"  {valid_count:3d}: sample={int(sample_idx):6d}, time={time_sec:7.2f}s, label={label}")
        valid_count += 1

    if save_report:
        with open(report_path, "w", encoding="utf-8") as file_obj:
            file_obj.write("TypeID report\n")
            file_obj.write(f"Sampling frequency: {sfreq} Hz\n")
            file_obj.write(f"Unique TypeID values: {np.unique(type_id)}\n")
            for id_num in sorted(type_map):
                file_obj.write(f"{id_num} -> {type_map[id_num]}\n")
        print(f"Saved report to: {report_path}")


def main():
    inspect_typeid_mapping()


if __name__ == "__main__":
    main()
