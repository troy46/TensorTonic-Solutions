def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    # Write code here
    _allowed_type_ = {"float":["int"]}
    def verify_single_record(record, schema):
        msg = []
        for column in schema:
            if column['column'] not in record:
                msg.append(f"{column['column']}: missing")
                continue
            value = record[column['column']]
            if value is None:
                if not column['nullable']:
                    msg.append(f"{column['column']}: null")
                continue
            if type(value).__name__!=column['type'] and (column['type'] not in _allowed_type_ or type(value).__name__ not in _allowed_type_[column['type']]):
                msg.append(f"{column['column']}: expected {column['type']}, got {type(value).__name__}")
                continue
            if ("max" in column and column["max"] is not None and value>column["max"]) or ("min" in column and column["min"] is not None and value<column["min"]):
                msg.append(f"{column['column']}: out of range")
        return not msg, msg

    check_result = [(i, *verify_single_record(record, schema)) for i,record in enumerate(records)]
    return check_result
                