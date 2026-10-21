"""Persist the API-edited FOC page with verified native and exported libraries."""
import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path


DOC_START = re.compile(br'(?m)(?=^\{"type":"DOCHEAD"\}\|\|)')
EXTRA_LIBRARIES = {
    "97ae5430dcdb9078": ("SYMBOL", "88f801fa73a24926a78b98f126429f07", "Netport-BI"),
    "924597b28086ca29": ("SYMBOL", "542d01b8b25e4236a453d7437c3944df", "Ground-GND"),
    "62d83f42dfd7ee79": ("DEVICE", "327808a2276743528035d8345c17831f", "Netport-BI"),
    "30d4be1084cd74d8": ("DEVICE", None, "Ground-GND"),
}


def split_documents(data):
    """Return each document's parsed header and untouched original bytes."""
    offset = 3 if data.startswith(b"\xef\xbb\xbf") else 0
    matches = list(DOC_START.finditer(data))
    if not matches or matches[0].start() != offset:
        raise ValueError("Document stream must begin with DOCHEAD")

    result = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(data)
        raw = data[start:end]
        first_line = raw.splitlines()[0]
        _, encoded_header = first_line.split(b"||", 1)
        header = json.loads(encoded_header.rstrip(b"|"))
        result.append((header, raw))
    return result


def record_payload(raw, record_type):
    for line in raw.splitlines():
        if b"||" not in line:
            continue
        try:
            key, value = line.split(b"||", 1)
            if json.loads(key).get("type") == record_type:
                return json.loads(value.rstrip(b"|"))
        except (UnicodeDecodeError, ValueError, TypeError):
            continue
    raise ValueError(f"Missing {record_type} record in document")


def extract_new_libraries(docs):
    by_uuid = {}
    for head, raw in docs:
        uuid = head.get("uuid")
        if uuid in EXTRA_LIBRARIES:
            by_uuid.setdefault(uuid, []).append((head, raw))

    additions = []
    for uuid, (expected_type, expected_source, expected_title) in EXTRA_LIBRARIES.items():
        matches = by_uuid.get(uuid, [])
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one exported {expected_type} {uuid}")
        head, raw = matches[0]
        if head.get("docType") != expected_type:
            raise ValueError(f"Unexpected document type for exported library {uuid}")
        meta = record_payload(raw, "META")
        if meta.get("title") != expected_title:
            raise ValueError(f"Unexpected library title for exported document {uuid}")
        source = meta.get("source", "").split("|", 1)[0]
        if expected_source is not None and source != expected_source:
            raise ValueError(f"Unexpected source ID for exported document {uuid}")
        if expected_type == "DEVICE":
            expected_symbol = "97ae5430dcdb9078" if uuid == "62d83f42dfd7ee79" else "924597b28086ca29"
            if meta.get("attributes", {}).get("Symbol") != expected_symbol:
                raise ValueError(f"Exported device {uuid} links to an unexpected symbol")
            if not source:
                raise ValueError(f"Exported device {uuid} is missing its source ID")
        additions.append(raw)
    return additions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-export", required=True)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    target = root / "eda工程文件/FOC_G431_Driver"
    if target.is_symlink() or not target.is_dir():
        raise ValueError("Wrong native target")

    archive_path = root / "归档/历史与调试.zip"
    baseline_entry = (
        "FOC驱动模块/EDA自动化/20261012标准连接符修订前/"
        "eda工程文件/FOC_G431_Driver_原理图电气修订/sch/Schematic1/P1.esch2"
    )
    with zipfile.ZipFile(archive_path) as archive:
        baseline = archive.read(baseline_entry)
    baseline_sha = hashlib.sha256(baseline).hexdigest().upper()
    if baseline_sha != "75AC862DD5C81FA4AF4B3211EC5E35AC5361CA7E3752430BBB023E04AEBF590D":
        raise ValueError(f"Unexpected SHA-256 for archived baseline: {baseline_sha}")

    baseline_docs = split_documents(baseline)
    library = [
        (head, raw)
        for head, raw in baseline_docs
        if head.get("docType") in {"FOOTPRINT", "SYMBOL", "DEVICE"}
    ]
    original_symbols = {head["uuid"] for head, _ in library if head.get("docType") == "SYMBOL"}
    original_devices = {head["uuid"] for head, _ in library if head.get("docType") == "DEVICE"}
    if original_symbols.intersection({"97ae5430dcdb9078", "924597b28086ca29"}):
        raise ValueError("An allowed added symbol already exists in the baseline")
    if original_devices.intersection({"62d83f42dfd7ee79", "30d4be1084cd74d8"}):
        raise ValueError("An allowed added device already exists in the baseline")

    with zipfile.ZipFile(args.project_export) as exported:
        bad_member = exported.testzip()
        if bad_member is not None:
            raise ValueError(f"Corrupt project export member: {bad_member}")
        sources = [name for name in exported.namelist() if name.endswith(".epru")]
        if len(sources) != 1:
            raise ValueError("Expected exactly one .epru document stream in the project export")
        export_docs = split_documents(exported.read(sources[0]))

    additions = extract_new_libraries(export_docs)
    pages = [(head, raw) for head, raw in export_docs if head.get("docType") == "SCH_PAGE"]
    if len(pages) != 1 or pages[0][0].get("uuid") != "90aa6c46493624cf":
        raise ValueError("Unexpected exported schematic page")

    head, page_raw = pages[0]
    baseline_symbols = original_symbols | {"97ae5430dcdb9078", "924597b28086ca29"}
    footprints = {head["uuid"] for head, _ in library if head.get("docType") == "FOOTPRINT"}
    for line in page_raw.splitlines():
        if b"||" not in line:
            continue
        try:
            key, value = line.split(b"||", 1)
            key = json.loads(key)
            value = json.loads(value.rstrip(b"|"))
        except (UnicodeDecodeError, ValueError, TypeError):
            continue
        if key.get("type") != "ATTR":
            continue
        if value.get("key") == "Symbol" and value.get("value"):
            if value["value"] not in baseline_symbols:
                raise ValueError(f"Page refers to an unverified symbol: {value['value']}")
        if value.get("key") == "Device" and value.get("value"):
            if value["value"] not in original_devices | {"62d83f42dfd7ee79", "30d4be1084cd74d8"}:
                raise ValueError(f"Page refers to an unverified device: {value['value']}")
        if value.get("key") == "Footprint" and value.get("value"):
            if value["value"] not in footprints:
                raise ValueError(f"Page refers to an unverified footprint: {value['value']}")

    stamp = int(time.time() * 1000)
    head.update(version=str(stamp), updateTime=stamp)
    first_line, separator, page_body = page_raw.partition(b"\n")
    if not separator:
        raise ValueError("Exported page has no body")
    newline = b"\r\n" if first_line.endswith(b"\r") else b"\n"
    page = (
        b'{"type":"DOCHEAD"}||'
        + json.dumps(head, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"|"
        + newline
        + page_body
    )
    native_page = target / "sch/Schematic1/P1.esch2"
    native_page.write_bytes(page + b"".join(raw for _, raw in library) + b"".join(additions))

    manifest_path = target / "FOC_G431_Driver.eprj3"
    manifest = json.loads(manifest_path.read_text("utf-8-sig"))
    manifest["content"] = "WeAct STM32G431插接核心板；单页FOC主板；COMP4监测SAFE欠压。PCB由人类设计。"
    manifest["profile"]["sheets"][head["uuid"]].update(version=head["version"], updateTime=stamp)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    files = [manifest_path, native_page, target / "sch/Schematic1/Schematic1.ecfg"]
    print(json.dumps({
        "method": "API page export plus SHA-verified original libraries and four source-validated standard symbol/device documents",
        "nativeHashes": {
            str(file.relative_to(root)): hashlib.sha256(file.read_bytes()).hexdigest().upper()
            for file in files
        },
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
