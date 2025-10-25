import os
from pathlib import Path
from scapy.all import rdpcap, IP, TCP, UDP
import pandas as pd
import numpy as np
from collections import defaultdict

print("Converting PCAPs to flow CSVs...")

pcap_dir = Path("data/raw/CICIDS2017")
output_dir = Path("data/raw/CICIDS2017")

pcap_files = list(pcap_dir.glob("*.pcap"))
print(f"Found {len(pcap_files)} PCAP files")

for pcap_file in pcap_files:
    print(f"\nProcessing: {pcap_file.name}")
    
    # Create CSV filename
    csv_name = pcap_file.stem + "_flows.csv"
    output_csv = output_dir / csv_name
    
    try:
        # Read PCAP
        packets = rdpcap(str(pcap_file))
        print(f"  Loaded {len(packets)} packets")
        
        # Extract basic flow features
        flows = []
        for pkt in packets:
            if IP in pkt:
                flow = {
                    'src_ip': pkt[IP].src,
                    'dst_ip': pkt[IP].dst,
                    'protocol': pkt[IP].proto,
                    'length': len(pkt),
                    'label': 'BENIGN'  # Default label
                }
                
                if TCP in pkt:
                    flow['src_port'] = pkt[TCP].sport
                    flow['dst_port'] = pkt[TCP].dport
                elif UDP in pkt:
                    flow['src_port'] = pkt[UDP].sport
                    flow['dst_port'] = pkt[UDP].dport
                else:
                    flow['src_port'] = 0
                    flow['dst_port'] = 0
                
                flows.append(flow)
        
        # Save to CSV
        df = pd.DataFrame(flows)
        df.to_csv(output_csv, index=False)
        print(f"  Saved: {output_csv.name} ({len(df)} flows)")
        
    except Exception as e:
        print(f"  Error: {e}")

print("\nDone! Now run: python run_complete_project.py")