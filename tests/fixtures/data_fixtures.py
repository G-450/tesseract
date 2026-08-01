import json


def create_synthetic_dataset(
    filepath, valid=True, missing_field=False, invalid_target=False, nan_target=False
):
    """Creates a synthetic JSON matching the dataset schema."""

    struct_data = {
        "@class": "Structure",
        "lattice": {"matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        "sites": [{"label": "H"}],
    }

    malformed_struct = {
        "@class": "NotStructure",
    }

    data = []
    if valid:
        data.append([struct_data, 1.5])
        data.append([struct_data, 2.5])

    if missing_field:
        data.append([struct_data])  # missing target

    if invalid_target:
        data.append([struct_data, "not-a-number"])

    if nan_target:
        data.append([struct_data, None])

    # Malformed structure
    if not valid and not missing_field and not invalid_target and not nan_target:
        data.append([malformed_struct, 0.0])

    dataset = {
        "index": list(range(len(data))),
        "columns": ["structure", "e_form"],
        "data": data,
    }

    # We must write NaN as literal NaN if we want JSON to fail,
    # but python json.dumps handles nan natively as NaN (which is technically invalid json, but parsers often handle it)
    # ijson might choke on unquoted NaN, so we might need to write it carefully.
    # Let's use simple dumps.
    with open(filepath, "w") as f:
        json.dump(dataset, f)
