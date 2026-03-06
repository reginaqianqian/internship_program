import argparse
import ast

from tactile_sdk.sharpa.tactile import Touch

key_type = {
    'version': str,
    'ip': str,
    'dest_ip': str,
    'stream_port': int,
    'exposure': int,
    'gain': int,
    'fps': int,
    'require_jpg': bool,
    'require_raw': bool,
    'require_deform': bool,
    'require_f6': bool,
    'require_infer': bool,
    'f6_offset': bytes,
    'deform_offset': bytes,
    'sn': str,
    'ftsid': str,
    'flash_exp': str,
    'refresh': str,
    'update_ini_img': str,
}

def main():
    parser = argparse.ArgumentParser(description='board configuration')
    parser.add_argument('-a', '--address', type=str, help='''
        ip e.g. 192.168.1.105
    ''', required=True)
    parser.add_argument('-c', '--channel', type=int, help='''
        channel e.g. 0
    ''', required=True)
    parser.add_argument('-k', '--key', type=str, help=f'''
        get/set key. key should be [{key_type.keys()}]
        when value is not given, it queries existing value from board
    ''', required=True)
    parser.add_argument('-v', '--value', type=str, help='''
        value to be set
        when value is not given, it perform get operation
    ''')
    args = parser.parse_args()

    if args.key not in key_type: raise RuntimeError(f'invalid key {args.key}')
    value = None
    if args.value is not None:
        try:
            if key_type[args.key] is bool:
                value = ast.literal_eval(args.value)
            else: value = key_type[args.key](args.value)
        except ValueError as e:
            print(f'invalid value "{args.value}" for {args.key}, should be {key_type[args.key]}')
            raise

    print(f"address: {args.address} channel: {args.channel} key: {args.key} value: {value}")
    touch = Touch('', -1, board_ip=[])
    res = touch.board_cfg(args.address, args.channel, args.key, value)
    print(f"response: {res}")

if __name__ == '__main__':
    main()
