import time

NUM_FAKE_PACKET = 2000

def run_fake_hw():
    import socket
    from scapy.all import rdpcap, UDP

    target_ip = '127.0.0.1'
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    pcap_file = 'test_py/cases/7.17.pcap'
    packets = rdpcap(pcap_file, NUM_FAKE_PACKET)

    start_send = time.perf_counter()
    first_packet_time = packets[0].time

    for packet in packets:
        if UDP in packet:
            dst_port = 50001
            payload = bytes(packet[UDP].payload)
            # simulate send time
            delay = packet.time - first_packet_time
            scheduled_time = start_send + delay
            sleep_time = scheduled_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(float(sleep_time) * 10)
            udp_socket.sendto(payload, (target_ip, dst_port))
    udp_socket.close()