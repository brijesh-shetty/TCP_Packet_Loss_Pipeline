#!/usr/bin/env python3
"""ecn_topology.py — Mininet Dumbbell Topology for Artificial ECN Experiments.

Creates the same network topology as the base paper (Welzl et al. 2024):
  - Bottleneck link with configurable bandwidth/delay
  - Background flows to create realistic congestion
  - RED queue with ECN on the bottleneck

KEY DESIGN:
  - Bottleneck (s1→s2): htb (rate limit) → RED+ECN (queue management)
  - Delay: netem on HOST interfaces (not bottleneck) to avoid buffer bloat
  - This ensures RED sees real queue buildup and properly marks ECN

tc on bottleneck:
  root (htb, handle 1:) rate=10mbit
    └── class 1:1
         └── RED+ECN (handle 10:)  ← we modify THIS for ECN injection

tc on hosts:
  root netem delay 15ms  (each direction = 30ms RTT total)
"""
import sys
import time
import re

from mininet.net import Mininet
from mininet.node import OVSBridge
from mininet.link import Link
from mininet.log import setLogLevel, info
from mininet.cli import CLI


def create_topology(bw_mbit=10, delay_ms=30, queue_size=100,
                    red_min=30000, red_max=90000, n_bg_flows=6):
    """Create dumbbell topology matching base paper.

    Topology (all hosts on 10.0.0.0/24):
        h1  (10.0.0.1)  ─┐                    ┌─ h3  (10.0.0.2)
        hs0 (10.0.0.10) ─┤                    ├─ hr0 (10.0.0.20)
        hs1 (10.0.0.11) ─┼── s1 ═══════ s2 ──┼─ hr1 (10.0.0.21)
        ...              ─┤   10Mbit           ├─ ...
        hs5 (10.0.0.15) ─┘   bottleneck       └─ hr5 (10.0.0.25)
    """
    net = Mininet(switch=OVSBridge, link=Link, controller=None)

    info('*** Creating topology\n')

    s1 = net.addSwitch('s1')
    s2 = net.addSwitch('s2')

    h1 = net.addHost('h1', ip='10.0.0.1/24')
    h3 = net.addHost('h3', ip='10.0.0.2/24')

    bg_senders = []
    bg_receivers = []
    for i in range(n_bg_flows):
        hs = net.addHost(f'hs{i}', ip=f'10.0.0.{10+i}/24')
        hr = net.addHost(f'hr{i}', ip=f'10.0.0.{20+i}/24')
        bg_senders.append(hs)
        bg_receivers.append(hr)

    # All plain links — no Mininet tc shaping
    net.addLink(h1, s1)
    for hs in bg_senders:
        net.addLink(hs, s1)
    net.addLink(s1, s2)  # bottleneck — we add tc manually
    net.addLink(h3, s2)
    for hr in bg_receivers:
        net.addLink(hr, s2)

    net.start()

    # --- TCP settings ---
    info('*** Configuring TCP settings\n')
    for host in [h1, h3] + bg_senders + bg_receivers:
        host.cmd('sysctl -w net.ipv4.tcp_ecn=1')
        host.cmd('sysctl -w net.ipv4.tcp_timestamps=1')
        host.cmd('sysctl -w net.ipv4.tcp_sack=1')

    # --- Add DELAY on host interfaces (not bottleneck) ---
    # Split delay evenly: half on sender side, half on receiver side
    half_delay = delay_ms // 2
    info(f'*** Adding {half_delay}ms delay on each host interface '
         f'(total RTT ≈ {delay_ms}ms)\n')

    for host in [h1] + bg_senders:
        intf = host.defaultIntf().name
        host.cmd(f'tc qdisc add dev {intf} root netem delay {half_delay}ms')

    for host in [h3] + bg_receivers:
        intf = host.defaultIntf().name
        host.cmd(f'tc qdisc add dev {intf} root netem delay {half_delay}ms')

    # --- Find bottleneck interfaces ---
    s1_to_s2_intf = None
    s2_to_s1_intf = None

    for intf in s1.intfList():
        link = intf.link
        if link:
            n1, n2 = link.intf1.node, link.intf2.node
            if (n1 == s1 and n2 == s2):
                s1_to_s2_intf = intf.name
                s2_to_s1_intf = link.intf2.name
            elif (n1 == s2 and n2 == s1):
                s1_to_s2_intf = link.intf2.name
                s2_to_s1_intf = link.intf1.name

    if not s1_to_s2_intf:
        info('  ✗ ERROR: Could not find bottleneck interface!\n')
        return net, {}

    info(f'*** Bottleneck: s1={s1_to_s2_intf}, s2={s2_to_s1_intf}\n')

    # --- Build tc on bottleneck: htb (rate limit) → RED+ECN ---
    # NO netem here — delay is on host interfaces
    info(f'*** Bottleneck tc: htb {bw_mbit}mbit → RED+ECN '
         f'(min={red_min}, max={red_max})\n')

    s2_node = net.get('s2')
    for switch, intf, direction in [(s1, s1_to_s2_intf, 's1→s2'),
                                     (s2_node, s2_to_s1_intf, 's2→s1')]:
        switch.cmd(f'tc qdisc del dev {intf} root 2>/dev/null')

        # htb root — rate limiting
        switch.cmd(f'tc qdisc add dev {intf} root handle 1: '
                   f'htb default 1 r2q 10')
        switch.cmd(f'tc class add dev {intf} parent 1: classid 1:1 '
                   f'htb rate {bw_mbit}mbit burst 15k')

        # RED with ECN — queue management (direct child of htb, NO netem)
        switch.cmd(f'tc qdisc add dev {intf} parent 1:1 handle 10: red '
                   f'limit 200000 min {red_min} max {red_max} '
                   f'avpkt 1000 bandwidth {bw_mbit}mbit ecn probability 0.1')

        info(f'  {direction} ({intf}): OK\n')

    # --- Verify ---
    tc_show = s1.cmd(f'tc qdisc show dev {s1_to_s2_intf}')
    info(f'*** tc on {s1_to_s2_intf}:\n')
    for line in tc_show.strip().split('\n'):
        info(f'    {line}\n')

    # Verify connectivity
    info('*** Connectivity check...\n')
    ping_result = h1.cmd(f'ping -c 3 -W 3 {h3.IP()}')
    rtt_match = re.search(r'rtt min/avg/max.*= ([\d.]+)/([\d.]+)/([\d.]+)',
                          ping_result)
    if rtt_match:
        info(f'  ✓ h1→h3 OK (min/avg/max RTT: '
             f'{rtt_match.group(1)}/{rtt_match.group(2)}/{rtt_match.group(3)} ms)\n')
    elif 'bytes from' in ping_result:
        info(f'  ✓ h1→h3 OK\n')
    else:
        info(f'  ✗ h1→h3 FAILED\n')
        info(f'    {ping_result}\n')

    # Quick bandwidth check
    info('*** Quick bandwidth check (5s)...\n')
    h3.cmd('iperf3 -s -D -p 9999')
    time.sleep(0.5)
    bw_result = h1.cmd(f'iperf3 -c {h3.IP()} -p 9999 -t 5 -f m')
    bw_match = re.search(r'(\d+\.?\d*)\s+Mbits/sec.*sender', bw_result)
    if bw_match:
        measured_bw = float(bw_match.group(1))
        info(f'  Measured: {measured_bw:.1f} Mbit/s '
             f'(expected ≈{bw_mbit} Mbit/s)\n')
    else:
        # Try to find any Mbits/sec
        bw_lines = [l for l in bw_result.split('\n') if 'Mbits/sec' in l]
        if bw_lines:
            info(f'  {bw_lines[-1].strip()}\n')
        else:
            info(f'  Could not parse bandwidth\n')
    h3.cmd('kill %iperf3 2>/dev/null; killall iperf3 2>/dev/null')
    time.sleep(0.5)

    return net, {
        'h1': h1, 'h3': h3,
        's1': s1, 's2': s2_node,
        'bg_senders': bg_senders,
        'bg_receivers': bg_receivers,
        's1_to_s2_intf': s1_to_s2_intf,
        's2_to_s1_intf': s2_to_s1_intf,
        'bw_mbit': bw_mbit,
        'delay_ms': delay_ms,
        'red_min': red_min,
        'red_max': red_max,
    }


def set_congestion_control(host, algo='cubic'):
    """Set TCP congestion control algorithm on a host."""
    host.cmd(f'sysctl -w net.ipv4.tcp_congestion_control={algo}')


def setup_baseline_qdisc(switch, intf_name, bw_mbit):
    """Replace RED with pfifo for baseline (keeps htb rate limit)."""
    switch.cmd(f'tc qdisc replace dev {intf_name} parent 1:1 handle 10: '
               f'pfifo limit 100')


def setup_red_ecn_qdisc(switch, intf_name, red_min, red_max, bw_mbit):
    """Replace pfifo with RED+ECN."""
    switch.cmd(f'tc qdisc replace dev {intf_name} parent 1:1 handle 10: red '
               f'limit 200000 min {red_min} max {red_max} '
               f'avpkt 1000 bandwidth {bw_mbit}mbit ecn probability 0.1')


def adjust_red_thresholds(switch, intf_name, new_min, new_max, bw_mbit=10):
    """Dynamically adjust RED thresholds — ECN injection mechanism.
    Changes handle 10: (the RED qdisc), leaving htb rate limit intact.
    """
    switch.cmd(f'tc qdisc change dev {intf_name} handle 10: red '
               f'limit 200000 min {new_min} max {new_max} '
               f'avpkt 1000 bandwidth {bw_mbit}mbit ecn probability 0.2')


def restore_red_thresholds(switch, intf_name, orig_min, orig_max, bw_mbit=10):
    """Restore original RED thresholds."""
    switch.cmd(f'tc qdisc change dev {intf_name} handle 10: red '
               f'limit 200000 min {orig_min} max {orig_max} '
               f'avpkt 1000 bandwidth {bw_mbit}mbit ecn probability 0.1')


if __name__ == '__main__':
    setLogLevel('info')

    if '--test' in sys.argv:
        print("Running connectivity + bottleneck test...")
        net, config = create_topology()
        if config:
            h1, h3 = config['h1'], config['h3']
            print("\n--- Full Ping Test ---")
            result = h1.cmd(f'ping -c 5 {h3.IP()}')
            print(result)

            print("\n--- Full Bandwidth Test (10s, should be ~10 Mbit) ---")
            h3.cmd('iperf3 -s -D')
            time.sleep(1)
            result = h1.cmd(f'iperf3 -c {h3.IP()} -t 10')
            print(result)

            h3.cmd('killall iperf3')
        net.stop()
        print("\n✓ Test complete!")
    else:
        net, config = create_topology()
        CLI(net)
        net.stop()
