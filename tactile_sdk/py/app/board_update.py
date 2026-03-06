import argparse

from tactile_sdk.sharpa.tactile import Touch

def main():
    parser = argparse.ArgumentParser(description='board configuration')
    parser.add_argument('-a', '--address', type=str, help='''
        ip e.g. 192.168.1.105
    ''', required=True)
    parser.add_argument('-f', '--firmware', type=str, help='''
        path of firmware to use
    ''', required=True)
    args = parser.parse_args()

    print(f"address: {args.address} firmware: {args.firmware}")
    touch = Touch('', -1, board_ip=[])
    res = touch.board_update(args.address, args.firmware)
    print(f"response: {res}")

if __name__ == '__main__':
    main()
