import os
from cicflowmeter.sniffer import create_sniffer

def extract_from_pcap(input_pcap, output_csv):
    """
    Extract flow features from a PCAP file using Python CICFlowMeter.
    """
    if not os.path.exists(input_pcap):
        raise FileNotFoundError(f"PCAP file not found: {input_pcap}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Processing PCAP: {input_pcap}")
    print(f"Output CSV: {output_csv}")
    
    # create_sniffer returns a tuple: (sniffer, output_writer)
    result = create_sniffer(
        input_file=input_pcap,
        input_interface=None,
        output_mode='csv',
        output=output_csv
    )
    
    # Unpack the tuple
    if isinstance(result, tuple):
        sniffer = result[0]
    else:
        sniffer = result
    
    sniffer.start()
    print(f"✅ Features extracted → {output_csv}")

def live_capture(interface="Ethernet", output_csv="data/flows/live_flows.csv"):
    """
    Capture live traffic and generate flow features in real time.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    print(f"Starting live capture on interface: {interface}")
    print(f"Output CSV: {output_csv}")
    
    sniffer = create_sniffer(
        input_file=None,
        input_interface=interface,
        output_mode="flow",
        output=output_csv
    )
    
    sniffer.start()
    print(f"📡 Live capture started on {interface}, writing to {output_csv}")
    print("Press Ctrl+C to stop")
    
    # Keep running
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n✓ Capture stopped")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Extract from PCAP: python src/feature_extractor.py pcap <input.pcap> <output.csv>")
        print("  Live capture: python src/feature_extractor.py live [interface] [output_dir]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == "pcap":
        input_pcap = sys.argv[2] if len(sys.argv) > 2 else "input.pcap"
        output_csv = sys.argv[3] if len(sys.argv) > 3 else "data/flows/pcap_flows.csv"
        extract_from_pcap(input_pcap, output_csv)
    
    elif mode == "live":
        interface = sys.argv[2] if len(sys.argv) > 2 else "Ethernet"
        output_csv = sys.argv[3] if len(sys.argv) > 3 else "data/flows/live_flows.csv"
        live_capture(interface, output_csv)
    
    else:
        print(f"Unknown mode: {mode}")