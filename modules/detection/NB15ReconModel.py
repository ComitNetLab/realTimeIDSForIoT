import json 

recon_column_names = [
    'dur', 'proto', 'state', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
    'sload', 'dload', 'sinpkt', 'dinpkt', 'sjit', 'djit', 'stcpb', 'dtcpb',
    'synack', 'ackdat', 'smean', 'dmean', 'trans_depth', 
    'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_ftp_cmd'
]

def map_payload_to_test_object(payload: str) -> dict:
    values = payload.split(',')
    converted_values = []
    for column, value in zip(recon_column_names, values):
        clean_value = value.strip()
        try:
            converted_value = int(clean_value)
        except ValueError:
            try:
                converted_value = float(clean_value)
            except ValueError:
                converted_value = clean_value.strip('"').strip("'")
        converted_values.append(converted_value)
    #for now assume that the data cames from csv format 
    return {column: value for column, value in zip(recon_column_names, converted_values)}

def recon_model(payload: str, model: any) -> None:
    #print('recon_model', payload)
    test_object = map_payload_to_test_object(payload)
    #print(json.dumps(test_object, indent=4))
    #model.predict(test_object)