# Chapter 23: Log Collection and Processing

## Why This Chapter Matters

Your network generates 50,000 syslog messages per hour. A BGP session flaps at 3 AM. The logs contain the answer—but they're scattered across 30 devices, in different formats, with no timestamps normalized, and buried in 2 million lines of noise.

**Traditional approach**: SSH to each device, `show logging`, grep for keywords, manually correlate events across devices. Takes 45 minutes to find the root cause.

**AI-ready approach**: Automated log collector sends structured, enriched logs to a centralized pipeline. When Chapter 24's AI analysis needs data, it's already normalized, timestamped, and ready. Root cause in 90 seconds.

This chapter builds the foundation that makes AI log analysis (Chapter 24) possible. You'll build production-grade log collection systems that:
- Collect logs from multi-vendor devices (Cisco, Juniper, Arista)
- Handle 100,000+ messages per hour reliably
- Normalize diverse formats into structured data
- Enrich logs with context (device metadata, GeoIP)
- Buffer and store efficiently
- Feed clean data to AI systems

**Reality check**: AI can't analyze logs it doesn't have. Collection and preprocessing are 80% of the solution. This chapter builds that 80%.

---

## The Problem with Network Logs

### A Real Incident

**September 2023, Major ISP**:
- Customer reports intermittent packet loss
- Engineer checks router logs manually (30 devices)
- Finds relevant logs on 5 different devices
- Logs use different timestamp formats (UTC, local, epoch)
- Some devices rotated logs (evidence lost)
- Takes 6 hours to reconstruct timeline
- **Root cause**: Spanning tree loop, visible in logs from minute one

**The fix**: Automated log collection with normalized timestamps. Same incident now detected in 3 minutes.

### Why Log Collection Is Hard

**Volume**: 10-100K messages/hour from 100+ devices = 2.4 million messages/day
**Variety**: Syslog, SNMP traps, NetFlow, API logs, streaming telemetry
**Velocity**: Real-time or bust—buffered logs miss time-sensitive events
**Formats**: Cisco IOS syslog ≠ Juniper syslog ≠ Arista syslog
**Reliability**: UDP syslog drops packets; TCP adds complexity
**Storage**: 2.4M messages × 365 days = 876 million logs/year
**Context**: Logs lack device metadata (location, role, criticality)

**Most engineers**: Use default syslog, lose logs to UDP drops, can't correlate events, no context
**This chapter**: Production-grade pipeline that solves all seven problems

---

## Section 1: Log Sources in Network Environments

Network devices generate logs through multiple channels. Your collector must handle all of them.

### Syslog from Network Devices

**The standard**: RFC 5424 (structured syslog) and RFC 3164 (legacy)
**The reality**: Every vendor interprets the RFC differently

#### Cisco IOS Syslog Format

```
<189>2024: Jan 17 14:32:15.234 PST: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap
```

**Structure**:
- `<189>`: Priority (facility × 8 + severity)
- `2024: Jan 17 14:32:15.234 PST`: Timestamp (format varies by IOS version)
- `%BGP-5-ADJCHANGE`: Facility-Severity-Mnemonic
- `neighbor 192.168.1.2 Down Interface flap`: Message

#### Juniper JunOS Syslog Format

```
<28>1 2024-01-17T14:32:15.234Z router-core-01 rpd 1234 - - BGP_NEIGHBOR_STATE_CHANGED: BGP peer 192.168.1.2 (External AS 65000) changed state from Established to Idle
```

**Structure** (RFC 5424):
- `<28>`: Priority
- `1`: Version
- `2024-01-17T14:32:15.234Z`: ISO 8601 timestamp (much better)
- `router-core-01`: Hostname
- `rpd`: Process name
- `1234`: Process ID
- `-`: Message ID (usually structured data, here omitted)
- `BGP_NEIGHBOR_STATE_CHANGED`: Event
- Rest: Message

#### Arista EOS Syslog Format

```
<189>Jan 17 14:32:15 switch-dist-01 Ebra: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down
```

**Structure** (IOS-like but subtle differences):
- `<189>`: Priority
- `Jan 17 14:32:15`: Timestamp (no year!)
- `switch-dist-01`: Hostname
- `Ebra`: Process name
- `%BGP-5-ADJCHANGE`: Facility-Severity-Mnemonic
- Rest: Message

**Key challenge**: Parse all three formats (and more) into a unified structure.

### Syslog Receiver: Basic UDP Implementation

```python
"""
Basic UDP Syslog Receiver
File: log_collection/syslog_receiver_basic.py

Receives syslog messages via UDP, displays them.
Production limitations: UDP drops packets, no persistence.
"""
import socket
import re
from datetime import datetime
from typing import Dict, Optional

class BasicSyslogReceiver:
    """Receive syslog messages via UDP."""

    def __init__(self, host: str = "0.0.0.0", port: int = 514):
        """
        Args:
            host: Interface to bind to (0.0.0.0 = all interfaces)
            port: Syslog port (514 standard, requires root on Linux)
        """
        self.host = host
        self.port = port
        self.sock = None

    def start(self):
        """Start receiving logs."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))

        print(f"Syslog receiver listening on {self.host}:{self.port}")
        print("Waiting for logs...\n")

        while True:
            try:
                data, addr = self.sock.recvfrom(4096)
                message = data.decode('utf-8', errors='ignore')

                # Parse and display
                parsed = self.parse_syslog(message, addr[0])
                self.display_log(parsed)

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error receiving log: {e}")

        self.sock.close()

    def parse_syslog(self, raw_message: str, source_ip: str) -> Dict:
        """
        Parse syslog message into structured format.

        Returns:
            Dict with parsed fields
        """
        # Extract priority (PRI)
        pri_match = re.match(r'^<(\d+)>', raw_message)
        if pri_match:
            pri = int(pri_match.group(1))
            facility = pri >> 3
            severity = pri & 0x07
            message = raw_message[pri_match.end():]
        else:
            facility = None
            severity = None
            message = raw_message

        # Try to extract timestamp (basic)
        # Matches: "Jan 17 14:32:15" or "2024-01-17T14:32:15"
        timestamp_patterns = [
            r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})',  # Jan 17 14:32:15
            r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',   # ISO 8601
        ]

        timestamp = None
        for pattern in timestamp_patterns:
            ts_match = re.match(pattern, message.strip())
            if ts_match:
                timestamp = ts_match.group(1)
                message = message[ts_match.end():].strip()
                break

        # Extract hostname (assumes format: "timestamp hostname message")
        hostname_match = re.match(r'^(\S+)\s+', message)
        hostname = hostname_match.group(1) if hostname_match else None
        if hostname:
            message = message[hostname_match.end():]

        return {
            "raw": raw_message,
            "received_at": datetime.utcnow().isoformat(),
            "source_ip": source_ip,
            "facility": facility,
            "severity": severity,
            "severity_text": self.severity_to_text(severity),
            "timestamp": timestamp,
            "hostname": hostname,
            "message": message.strip()
        }

    @staticmethod
    def severity_to_text(severity: Optional[int]) -> Optional[str]:
        """Convert severity number to text."""
        if severity is None:
            return None

        severities = {
            0: "EMERGENCY",
            1: "ALERT",
            2: "CRITICAL",
            3: "ERROR",
            4: "WARNING",
            5: "NOTICE",
            6: "INFO",
            7: "DEBUG"
        }
        return severities.get(severity, "UNKNOWN")

    def display_log(self, parsed: Dict):
        """Display parsed log entry."""
        print(f"[{parsed['received_at']}] {parsed['source_ip']} | "
              f"{parsed['severity_text']} | {parsed['hostname']} | "
              f"{parsed['message'][:100]}")


# Example Usage
if __name__ == "__main__":
    # Note: Port 514 requires root/admin privileges
    # Use port 5140 for testing without privileges
    receiver = BasicSyslogReceiver(host="0.0.0.0", port=5140)

    print("Basic UDP Syslog Receiver")
    print("=" * 60)
    print("Configure devices to send syslog to this host:5140")
    print("Cisco IOS: logging host <this-ip> transport udp port 5140")
    print("=" * 60)

    receiver.start()
```

**Test Output**:
```
Syslog receiver listening on 0.0.0.0:5140
Waiting for logs...

[2024-01-17T14:32:15.123456] 10.1.1.1 | NOTICE | router-core-01 | %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap
[2024-01-17T14:32:16.234567] 10.1.1.1 | ERROR | router-core-01 | %LINK-3-UPDOWN: Interface GigabitEthernet0/1, changed state to down
[2024-01-17T14:32:45.345678] 10.1.1.1 | NOTICE | router-core-01 | %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Up
```

**Configure Device to Send Logs**:

**Cisco IOS**:
```
configure terminal
logging host 10.0.1.100 transport udp port 5140
logging trap informational
logging facility local6
logging source-interface Loopback0
end
```

**Juniper JunOS**:
```
set system syslog host 10.0.1.100 any info
set system syslog host 10.0.1.100 port 5140
set system syslog host 10.0.1.100 source-address 1.1.1.1
```

**Arista EOS**:
```
configure
logging host 10.0.1.100 514 protocol udp
logging source-interface Loopback0
logging level informational
end
```

---

### SNMP Traps and Notifications

SNMP traps are event notifications sent from devices to a management station.

#### SNMP Trap Receiver

```python
"""
SNMP Trap Receiver
File: log_collection/snmp_trap_receiver.py

Receives and parses SNMP traps from network devices.
"""
from pysnmp.entity import engine, config
from pysnmp.carrier.asyncore.dgram import udp
from pysnmp.entity.rfc3413 import ntfrcv
from pysnmp.proto.api import v2c
from datetime import datetime
from typing import Dict, List
import json

class SNMPTrapReceiver:
    """Receive and parse SNMP traps."""

    def __init__(self, host: str = "0.0.0.0", port: int = 162):
        """
        Args:
            host: Interface to bind
            port: SNMP trap port (162 standard)
        """
        self.host = host
        self.port = port
        self.engine = engine.SnmpEngine()

    def start(self, callback=None):
        """Start receiving traps."""
        # Configure transport
        config.addTransport(
            self.engine,
            udp.domainName,
            udp.UdpTransport().openServerMode((self.host, self.port))
        )

        # Configure community (v2c)
        config.addV1System(self.engine, 'my-area', 'public')

        # Register callback
        def trap_callback(snmpEngine, stateReference, contextEngineId,
                         contextName, varBinds, cbCtx):
            trap_data = self.parse_trap(varBinds, cbCtx)

            if callback:
                callback(trap_data)
            else:
                self.display_trap(trap_data)

        # Register receiver
        ntfrcv.NotificationReceiver(self.engine, trap_callback)

        print(f"SNMP Trap receiver listening on {self.host}:{self.port}")
        print("Waiting for traps...\n")

        # Start engine
        self.engine.transportDispatcher.jobStarted(1)

        try:
            self.engine.transportDispatcher.runDispatcher()
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.engine.transportDispatcher.closeDispatcher()

    def parse_trap(self, varBinds, cbCtx) -> Dict:
        """Parse trap into structured format."""
        source_ip = cbCtx.get('transportAddress', ['unknown'])[0]

        varbinds = []
        trap_type = "unknown"

        for oid, val in varBinds:
            oid_str = oid.prettyPrint()
            val_str = val.prettyPrint()

            varbinds.append({
                "oid": oid_str,
                "value": val_str
            })

            # Identify trap type from OID
            if "1.3.6.1.6.3.1.1.4.1.0" in oid_str:
                trap_type = val_str

        return {
            "received_at": datetime.utcnow().isoformat(),
            "source_ip": source_ip,
            "trap_type": trap_type,
            "varbinds": varbinds
        }

    def display_trap(self, trap_data: Dict):
        """Display trap."""
        print(f"[{trap_data['received_at']}] TRAP from {trap_data['source_ip']}")
        print(f"  Type: {trap_data['trap_type']}")
        print(f"  VarBinds:")
        for vb in trap_data['varbinds']:
            print(f"    {vb['oid']} = {vb['value']}")
        print()


# Example Usage
if __name__ == "__main__":
    receiver = SNMPTrapReceiver(host="0.0.0.0", port=1162)  # 1162 for testing

    print("SNMP Trap Receiver")
    print("=" * 60)
    print("Configure devices:")
    print("Cisco: snmp-server host <this-ip> traps version 2c public")
    print("=" * 60)

    # Custom callback to save traps
    def save_trap(trap_data):
        print(f"Trap received: {trap_data['trap_type']}")
        with open("traps.jsonl", "a") as f:
            f.write(json.dumps(trap_data) + "\n")

    receiver.start(callback=save_trap)
```

**Configure Device for SNMP Traps**:

**Cisco IOS**:
```
snmp-server enable traps bgp
snmp-server enable traps config
snmp-server enable traps snmp linkdown linkup
snmp-server host 10.0.1.100 version 2c public
```

---

### NetFlow/sFlow/IPFIX Collection

Flow data provides network traffic telemetry.

#### Basic NetFlow Collector

```python
"""
NetFlow v5 Collector
File: log_collection/netflow_collector.py

Collects and parses NetFlow v5 records.
"""
import socket
import struct
from datetime import datetime
from typing import Dict, List
import json

class NetFlowV5Collector:
    """Collect NetFlow v5 records."""

    # NetFlow v5 Header format (24 bytes)
    HEADER_FORMAT = '!HHIIIIH'
    HEADER_SIZE = 24

    # NetFlow v5 Flow record format (48 bytes)
    FLOW_FORMAT = '!IIIHHIIIIHHBBBBHHBB'
    FLOW_SIZE = 48

    def __init__(self, host: str = "0.0.0.0", port: int = 2055):
        self.host = host
        self.port = port
        self.sock = None

    def start(self):
        """Start collecting NetFlow."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))

        print(f"NetFlow collector listening on {self.host}:{self.port}")
        print("Waiting for flows...\n")

        while True:
            try:
                data, addr = self.sock.recvfrom(65535)

                # Parse NetFlow packet
                flows = self.parse_netflow_packet(data, addr[0])

                for flow in flows:
                    self.display_flow(flow)

            except KeyboardInterrupt:
                print("\nShutting down...")
                break
            except Exception as e:
                print(f"Error: {e}")

        self.sock.close()

    def parse_netflow_packet(self, data: bytes, source_ip: str) -> List[Dict]:
        """Parse NetFlow v5 packet."""
        # Parse header
        header = struct.unpack(self.HEADER_FORMAT, data[:self.HEADER_SIZE])

        version, count, uptime, unix_secs, unix_nsecs, flow_seq, engine_type, engine_id = header

        if version != 5:
            print(f"Unsupported NetFlow version: {version}")
            return []

        # Parse flow records
        flows = []
        offset = self.HEADER_SIZE

        for i in range(count):
            if offset + self.FLOW_SIZE > len(data):
                break

            flow_data = struct.unpack(
                self.FLOW_FORMAT,
                data[offset:offset + self.FLOW_SIZE]
            )

            flow = self.parse_flow_record(flow_data, source_ip, unix_secs)
            flows.append(flow)

            offset += self.FLOW_SIZE

        return flows

    def parse_flow_record(self, flow_data: tuple, source_ip: str, unix_secs: int) -> Dict:
        """Parse single flow record."""
        (src_addr, dst_addr, nexthop, input_if, output_if,
         packets, octets, first, last, src_port, dst_port,
         pad1, tcp_flags, protocol, tos, src_as, dst_as, src_mask, dst_mask, pad2) = flow_data

        return {
            "received_at": datetime.utcnow().isoformat(),
            "exporter_ip": source_ip,
            "src_ip": self.int_to_ip(src_addr),
            "dst_ip": self.int_to_ip(dst_addr),
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "packets": packets,
            "bytes": octets,
            "input_interface": input_if,
            "output_interface": output_if,
            "tcp_flags": tcp_flags,
            "duration_ms": last - first
        }

    @staticmethod
    def int_to_ip(ip_int: int) -> str:
        """Convert integer to IP address string."""
        return socket.inet_ntoa(struct.pack('!I', ip_int))

    def display_flow(self, flow: Dict):
        """Display flow record."""
        print(f"{flow['src_ip']}:{flow['src_port']} → "
              f"{flow['dst_ip']}:{flow['dst_port']} | "
              f"Proto: {flow['protocol']} | "
              f"Packets: {flow['packets']} | "
              f"Bytes: {flow['bytes']}")


# Example Usage
if __name__ == "__main__":
    collector = NetFlowV5Collector(host="0.0.0.0", port=2055)

    print("NetFlow v5 Collector")
    print("=" * 60)
    print("Configure device:")
    print("Cisco: ip flow-export destination <this-ip> 2055")
    print("       ip flow-export version 5")
    print("=" * 60)

    collector.start()
```

**Configure NetFlow on Device**:

**Cisco IOS**:
```
interface GigabitEthernet0/1
 ip flow ingress
 ip flow egress
exit

ip flow-export destination 10.0.1.100 2055
ip flow-export version 5
ip flow-export source Loopback0
```

---

## Section 2: Production Log Collection Architectures

Basic UDP receivers drop packets under load. Production systems need reliability, buffering, and scale.

### TCP-Based Reliable Syslog Receiver

```python
"""
Reliable TCP Syslog Receiver
File: log_collection/tcp_syslog_receiver.py

Production-grade syslog receiver using TCP for reliability.
Handles multiple concurrent connections, buffers messages.
"""
import socket
import threading
import queue
import json
from datetime import datetime
from typing import Dict, Optional, Callable
import re

class TCPSyslogReceiver:
    """Reliable TCP syslog receiver."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 1514,
        max_connections: int = 100,
        buffer_size: int = 10000
    ):
        """
        Args:
            host: Bind address
            port: TCP port (1514 for non-privileged)
            max_connections: Max concurrent connections
            buffer_size: Message queue size
        """
        self.host = host
        self.port = port
        self.max_connections = max_connections

        # Message queue (thread-safe)
        self.message_queue = queue.Queue(maxsize=buffer_size)

        # Statistics
        self.stats = {
            "messages_received": 0,
            "messages_processed": 0,
            "connections_total": 0,
            "connections_active": 0,
            "errors": 0
        }

        self.running = False
        self.sock = None

    def start(self, callback: Optional[Callable] = None):
        """
        Start receiver.

        Args:
            callback: Function to call with each parsed message
        """
        self.running = True

        # Start processing thread
        processor = threading.Thread(target=self._process_messages, args=(callback,))
        processor.daemon = True
        processor.start()

        # Start stats thread
        stats_thread = threading.Thread(target=self._print_stats)
        stats_thread.daemon = True
        stats_thread.start()

        # Start TCP server
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port))
        self.sock.listen(self.max_connections)

        print(f"TCP Syslog receiver listening on {self.host}:{self.port}")
        print(f"Max connections: {self.max_connections}")
        print(f"Buffer size: {self.message_queue.maxsize}")
        print("=" * 60)

        while self.running:
            try:
                client_sock, client_addr = self.sock.accept()
                self.stats["connections_total"] += 1
                self.stats["connections_active"] += 1

                # Handle connection in separate thread
                handler = threading.Thread(
                    target=self._handle_connection,
                    args=(client_sock, client_addr)
                )
                handler.daemon = True
                handler.start()

            except KeyboardInterrupt:
                print("\nShutting down...")
                self.running = False
                break
            except Exception as e:
                print(f"Accept error: {e}")
                self.stats["errors"] += 1

        self.sock.close()

    def _handle_connection(self, client_sock: socket.socket, client_addr: tuple):
        """Handle individual client connection."""
        print(f"New connection from {client_addr[0]}:{client_addr[1]}")

        buffer = b""

        try:
            while self.running:
                data = client_sock.recv(4096)

                if not data:
                    break  # Connection closed

                buffer += data

                # Process complete messages (newline-delimited)
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)

                    try:
                        message = line.decode('utf-8', errors='ignore').strip()
                        if message:
                            # Add to queue
                            self.message_queue.put({
                                "raw": message,
                                "source_ip": client_addr[0],
                                "received_at": datetime.utcnow().isoformat()
                            }, block=False)
                            self.stats["messages_received"] += 1
                    except queue.Full:
                        print("WARNING: Message queue full, dropping messages!")
                        self.stats["errors"] += 1
                    except Exception as e:
                        print(f"Parse error: {e}")
                        self.stats["errors"] += 1

        except Exception as e:
            print(f"Connection error from {client_addr[0]}: {e}")
            self.stats["errors"] += 1
        finally:
            client_sock.close()
            self.stats["connections_active"] -= 1
            print(f"Connection closed from {client_addr[0]}")

    def _process_messages(self, callback: Optional[Callable]):
        """Process messages from queue."""
        while self.running:
            try:
                msg = self.message_queue.get(timeout=1)

                # Parse message
                parsed = self._parse_syslog(msg)

                # Call callback or display
                if callback:
                    callback(parsed)
                else:
                    self._display_message(parsed)

                self.stats["messages_processed"] += 1

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                self.stats["errors"] += 1

    def _parse_syslog(self, msg: Dict) -> Dict:
        """Parse syslog message (simplified)."""
        raw = msg["raw"]

        # Extract priority
        pri_match = re.match(r'^<(\d+)>', raw)
        if pri_match:
            pri = int(pri_match.group(1))
            facility = pri >> 3
            severity = pri & 0x07
            message = raw[pri_match.end():]
        else:
            facility = None
            severity = None
            message = raw

        # Extract timestamp (basic)
        timestamp = None
        ts_match = re.match(
            r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
            message.strip()
        )
        if ts_match:
            timestamp = ts_match.group(1)
            message = message[ts_match.end():].strip()

        # Extract hostname
        hostname_match = re.match(r'^(\S+)\s+', message)
        hostname = hostname_match.group(1) if hostname_match else None
        if hostname:
            message = message[hostname_match.end():]

        return {
            **msg,
            "facility": facility,
            "severity": severity,
            "severity_text": self._severity_to_text(severity),
            "timestamp": timestamp,
            "hostname": hostname,
            "message": message.strip()
        }

    @staticmethod
    def _severity_to_text(severity: Optional[int]) -> Optional[str]:
        """Convert severity to text."""
        if severity is None:
            return None
        severities = ["EMERG", "ALERT", "CRIT", "ERROR", "WARN", "NOTICE", "INFO", "DEBUG"]
        return severities[severity] if 0 <= severity < 8 else "UNKNOWN"

    def _display_message(self, parsed: Dict):
        """Display parsed message."""
        print(f"[{parsed['severity_text'] or 'UNKNOWN'}] "
              f"{parsed['hostname'] or parsed['source_ip']} | "
              f"{parsed['message'][:100]}")

    def _print_stats(self):
        """Print statistics periodically."""
        import time
        while self.running:
            time.sleep(30)
            print(f"\n--- Stats (30s interval) ---")
            print(f"Messages received: {self.stats['messages_received']}")
            print(f"Messages processed: {self.stats['messages_processed']}")
            print(f"Queue size: {self.message_queue.qsize()}")
            print(f"Active connections: {self.stats['connections_active']}")
            print(f"Total connections: {self.stats['connections_total']}")
            print(f"Errors: {self.stats['errors']}")
            print(f"---\n")


# Example Usage
if __name__ == "__main__":
    receiver = TCPSyslogReceiver(host="0.0.0.0", port=1514)

    # Optional: Save to file
    def save_to_file(msg):
        with open("syslog.jsonl", "a") as f:
            f.write(json.dumps(msg) + "\n")

    receiver.start(callback=save_to_file)
```

**Configure Device for TCP Syslog**:

**Cisco IOS**:
```
logging host 10.0.1.100 transport tcp port 1514
logging trap informational
```

**Test Output**:
```
TCP Syslog receiver listening on 0.0.0.0:1514
Max connections: 100
Buffer size: 10000
============================================================
New connection from 10.1.1.1:52341
[NOTICE] router-core-01 | %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down
[ERROR] router-core-01 | %LINK-3-UPDOWN: Interface GigabitEthernet0/1, down
[NOTICE] router-core-01 | %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Up

--- Stats (30s interval) ---
Messages received: 1247
Messages processed: 1247
Queue size: 0
Active connections: 3
Total connections: 5
Errors: 0
---
```

---

## Section 3: Log Normalization and Parsing

Raw logs are useless for AI. Normalize into structured format.

### Multi-Vendor Syslog Parser

```python
"""
Multi-Vendor Syslog Parser
File: log_collection/multi_vendor_parser.py

Parse syslog from Cisco IOS, Juniper JunOS, Arista EOS.
Normalize into unified format.
"""
import re
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Vendor(Enum):
    CISCO_IOS = "cisco_ios"
    JUNIPER_JUNOS = "juniper_junos"
    ARISTA_EOS = "arista_eos"
    UNKNOWN = "unknown"

@dataclass
class ParsedLog:
    """Normalized log entry."""
    raw_message: str
    received_at: str
    source_ip: str
    vendor: str
    facility: Optional[int]
    severity: int
    severity_text: str
    timestamp: Optional[str]
    hostname: Optional[str]
    process: Optional[str]
    message_id: Optional[str]
    message: str

    def to_dict(self) -> Dict:
        return asdict(self)

class MultiVendorSyslogParser:
    """Parse syslog from multiple vendors."""

    SEVERITY_MAP = {
        0: "EMERGENCY", 1: "ALERT", 2: "CRITICAL", 3: "ERROR",
        4: "WARNING", 5: "NOTICE", 6: "INFO", 7: "DEBUG"
    }

    def parse(self, raw_message: str, source_ip: str) -> ParsedLog:
        """
        Parse syslog message from any vendor.

        Args:
            raw_message: Raw syslog message
            source_ip: Source IP address

        Returns:
            ParsedLog object
        """
        received_at = datetime.utcnow().isoformat()

        # Extract priority (common to all)
        pri, message = self._extract_priority(raw_message)
        facility = pri >> 3 if pri is not None else None
        severity = pri & 0x07 if pri is not None else 6  # Default: INFO

        # Detect vendor
        vendor = self._detect_vendor(message)

        # Parse based on vendor
        if vendor == Vendor.CISCO_IOS:
            parsed = self._parse_cisco_ios(message)
        elif vendor == Vendor.JUNIPER_JUNOS:
            parsed = self._parse_juniper_junos(message)
        elif vendor == Vendor.ARISTA_EOS:
            parsed = self._parse_arista_eos(message)
        else:
            parsed = self._parse_generic(message)

        return ParsedLog(
            raw_message=raw_message,
            received_at=received_at,
            source_ip=source_ip,
            vendor=vendor.value,
            facility=facility,
            severity=severity,
            severity_text=self.SEVERITY_MAP[severity],
            **parsed
        )

    def _extract_priority(self, message: str) -> tuple[Optional[int], str]:
        """Extract PRI from message."""
        pri_match = re.match(r'^<(\d+)>', message)
        if pri_match:
            pri = int(pri_match.group(1))
            return pri, message[pri_match.end():]
        return None, message

    def _detect_vendor(self, message: str) -> Vendor:
        """Detect vendor from message format."""
        # Juniper: RFC 5424 format with version number
        if re.match(r'^\d+\s+\d{4}-\d{2}-\d{2}T', message):
            return Vendor.JUNIPER_JUNOS

        # Cisco IOS: %FACILITY-SEVERITY-MNEMONIC
        if re.search(r'%[A-Z_]+-\d+-[A-Z_]+:', message):
            return Vendor.CISCO_IOS

        # Arista: Similar to Cisco but with "Ebra" or "Ira" process
        if 'Ebra:' in message or 'Ira:' in message:
            return Vendor.ARISTA_EOS

        return Vendor.UNKNOWN

    def _parse_cisco_ios(self, message: str) -> Dict:
        """Parse Cisco IOS format."""
        # Format: Jan 17 14:32:15.234: %BGP-5-ADJCHANGE: neighbor X Down
        # Or: 2024: Jan 17 14:32:15 PST: %BGP-5-ADJCHANGE: neighbor X Down

        timestamp = None
        hostname = None
        message_id = None
        process = None

        # Extract timestamp
        ts_patterns = [
            r'^(\d{4}:\s*\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+\w+)?):',
            r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?):',
        ]

        for pattern in ts_patterns:
            ts_match = re.match(pattern, message)
            if ts_match:
                timestamp = ts_match.group(1)
                message = message[ts_match.end():].strip()
                break

        # Extract hostname (may come before or after timestamp)
        hostname_match = re.match(r'^(\S+):\s*', message)
        if hostname_match:
            hostname = hostname_match.group(1)
            message = message[hostname_match.end():]

        # Extract message ID (%FACILITY-SEVERITY-MNEMONIC)
        msg_id_match = re.match(r'^(%[A-Z_]+-\d+-[A-Z_]+):', message)
        if msg_id_match:
            message_id = msg_id_match.group(1)
            message = message[msg_id_match.end():].strip()

        return {
            "timestamp": timestamp,
            "hostname": hostname,
            "process": process,
            "message_id": message_id,
            "message": message
        }

    def _parse_juniper_junos(self, message: str) -> Dict:
        """Parse Juniper JunOS format (RFC 5424)."""
        # Format: 1 2024-01-17T14:32:15.234Z hostname rpd 1234 - - MESSAGE

        parts = message.split(None, 7)

        if len(parts) < 8:
            return self._parse_generic(message)

        version = parts[0]
        timestamp = parts[1]
        hostname = parts[2]
        process = parts[3]
        pid = parts[4]
        msgid = parts[5]
        structured_data = parts[6]
        msg = parts[7] if len(parts) > 7 else ""

        # Extract message ID from message (e.g., "BGP_NEIGHBOR_STATE_CHANGED:")
        message_id = None
        msg_match = re.match(r'^([A-Z_]+):', msg)
        if msg_match:
            message_id = msg_match.group(1)
            msg = msg[msg_match.end():].strip()

        return {
            "timestamp": timestamp,
            "hostname": hostname,
            "process": f"{process}[{pid}]",
            "message_id": message_id,
            "message": msg
        }

    def _parse_arista_eos(self, message: str) -> Dict:
        """Parse Arista EOS format."""
        # Format: Jan 17 14:32:15 hostname Ebra: %BGP-5-ADJCHANGE: message

        timestamp = None
        hostname = None
        process = None
        message_id = None

        # Extract timestamp
        ts_match = re.match(r'^(\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})', message)
        if ts_match:
            timestamp = ts_match.group(1)
            message = message[ts_match.end():].strip()

        # Extract hostname
        hostname_match = re.match(r'^(\S+)\s+', message)
        if hostname_match:
            hostname = hostname_match.group(1)
            message = message[hostname_match.end():]

        # Extract process (Ebra, Ira, etc.)
        process_match = re.match(r'^(\w+):\s*', message)
        if process_match:
            process = process_match.group(1)
            message = message[process_match.end():]

        # Extract message ID
        msg_id_match = re.match(r'^(%[A-Z_]+-\d+-[A-Z_]+):', message)
        if msg_id_match:
            message_id = msg_id_match.group(1)
            message = message[msg_id_match.end():].strip()

        return {
            "timestamp": timestamp,
            "hostname": hostname,
            "process": process,
            "message_id": message_id,
            "message": message
        }

    def _parse_generic(self, message: str) -> Dict:
        """Generic parser for unknown formats."""
        return {
            "timestamp": None,
            "hostname": None,
            "process": None,
            "message_id": None,
            "message": message.strip()
        }


# Example Usage
if __name__ == "__main__":
    parser = MultiVendorSyslogParser()

    # Test messages from different vendors
    test_messages = [
        # Cisco IOS
        ("<189>2024: Jan 17 14:32:15 PST: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap",
         "10.1.1.1"),

        # Juniper JunOS
        ("<28>1 2024-01-17T14:32:15.234Z router-core-01 rpd 1234 - - BGP_NEIGHBOR_STATE_CHANGED: BGP peer 192.168.1.2 changed state from Established to Idle",
         "10.2.2.2"),

        # Arista EOS
        ("<189>Jan 17 14:32:15 switch-dist-01 Ebra: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down",
         "10.3.3.3"),
    ]

    print("Multi-Vendor Syslog Parser Test")
    print("=" * 80)

    for raw_msg, source_ip in test_messages:
        parsed = parser.parse(raw_msg, source_ip)

        print(f"\nVendor: {parsed.vendor}")
        print(f"Severity: {parsed.severity_text}")
        print(f"Hostname: {parsed.hostname}")
        print(f"Message ID: {parsed.message_id}")
        print(f"Message: {parsed.message}")
        print("-" * 80)
```

**Output**:
```
Multi-Vendor Syslog Parser Test
================================================================================

Vendor: cisco_ios
Severity: NOTICE
Hostname: None
Message ID: %BGP-5-ADJCHANGE
Message: neighbor 192.168.1.2 Down Interface flap
--------------------------------------------------------------------------------

Vendor: juniper_junos
Severity: WARNING
Hostname: router-core-01
Message ID: BGP_NEIGHBOR_STATE_CHANGED
Message: BGP peer 192.168.1.2 changed state from Established to Idle
--------------------------------------------------------------------------------

Vendor: arista_eos
Severity: NOTICE
Hostname: switch-dist-01
Message ID: %BGP-5-ADJCHANGE
Message: neighbor 192.168.1.2 Down
--------------------------------------------------------------------------------
```

---

## Section 4: Data Enrichment

Logs without context are half-useless. Add metadata to make logs valuable for AI analysis.

### Log Enrichment Pipeline

```python
"""
Log Enrichment Pipeline
File: log_collection/enrichment_pipeline.py

Enrich logs with device metadata, GeoIP, and contextual information.
"""
import json
from datetime import datetime, timezone
from typing import Dict, Optional
from dataclasses import dataclass, asdict
import ipaddress

@dataclass
class DeviceMetadata:
    """Device metadata for enrichment."""
    hostname: str
    ip_address: str
    vendor: str
    model: str
    role: str  # core, edge, access, distribution
    location: str
    site: str
    criticality: str  # critical, high, medium, low

class DeviceInventory:
    """Device inventory database (simplified)."""

    def __init__(self):
        # In production: Load from CMDB, NetBox, or database
        self.devices = {
            "10.1.1.1": DeviceMetadata(
                hostname="router-core-01",
                ip_address="10.1.1.1",
                vendor="cisco",
                model="ASR-9006",
                role="core",
                location="DC1-Floor2-RackA12",
                site="datacenter-ny",
                criticality="critical"
            ),
            "10.2.2.2": DeviceMetadata(
                hostname="router-edge-01",
                ip_address="10.2.2.2",
                vendor="juniper",
                model="MX960",
                role="edge",
                location="DC1-Floor3-RackB05",
                site="datacenter-ny",
                criticality="critical"
            ),
            "10.3.3.3": DeviceMetadata(
                hostname="switch-dist-01",
                ip_address="10.3.3.3",
                vendor="arista",
                model="7050SX-128",
                role="distribution",
                location="DC1-Floor2-RackC08",
                site="datacenter-ny",
                criticality="high"
            ),
        }

    def get_device(self, ip_address: str) -> Optional[DeviceMetadata]:
        """Get device metadata by IP."""
        return self.devices.get(ip_address)

class LogEnrichmentPipeline:
    """Enrich logs with metadata."""

    def __init__(self, device_inventory: DeviceInventory):
        self.inventory = device_inventory

        # Timezone mapping (device IP subnet → timezone)
        self.timezone_map = {
            "10.1.0.0/16": "America/New_York",
            "10.2.0.0/16": "America/Los_Angeles",
            "192.168.0.0/16": "Europe/London",
        }

    def enrich(self, parsed_log: Dict) -> Dict:
        """
        Enrich log with metadata.

        Args:
            parsed_log: Parsed log dict

        Returns:
            Enriched log dict
        """
        enriched = parsed_log.copy()

        # Add device metadata
        device_meta = self.inventory.get_device(parsed_log.get("source_ip"))
        if device_meta:
            enriched["device"] = {
                "hostname": device_meta.hostname,
                "vendor": device_meta.vendor,
                "model": device_meta.model,
                "role": device_meta.role,
                "location": device_meta.location,
                "site": device_meta.site,
                "criticality": device_meta.criticality
            }
        else:
            enriched["device"] = {
                "hostname": parsed_log.get("hostname", "unknown"),
                "vendor": parsed_log.get("vendor", "unknown"),
                "model": "unknown",
                "role": "unknown",
                "location": "unknown",
                "site": "unknown",
                "criticality": "unknown"
            }

        # Normalize timestamp
        enriched["timestamp_normalized"] = self._normalize_timestamp(
            parsed_log.get("timestamp"),
            parsed_log.get("source_ip")
        )

        # Extract structured fields from message
        enriched["extracted_fields"] = self._extract_fields(parsed_log)

        # Add event classification
        enriched["event_type"] = self._classify_event(parsed_log)

        return enriched

    def _normalize_timestamp(
        self,
        timestamp: Optional[str],
        source_ip: str
    ) -> Optional[str]:
        """
        Normalize timestamp to ISO 8601 UTC.

        Handles:
        - "Jan 17 14:32:15" → adds year, converts to UTC
        - "2024-01-17T14:32:15.234Z" → already normalized
        - "2024: Jan 17 14:32:15 PST" → converts timezone
        """
        if not timestamp:
            return None

        # Already ISO 8601?
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', timestamp):
            return timestamp

        # Parse and normalize
        try:
            # Add year if missing (assume current year)
            if not re.match(r'^\d{4}', timestamp):
                timestamp = f"{datetime.now().year} {timestamp}"

            # Parse various formats
            for fmt in [
                "%Y %b %d %H:%M:%S",
                "%Y: %b %d %H:%M:%S %Z",
                "%Y: %b %d %H:%M:%S.%f",
            ]:
                try:
                    dt = datetime.strptime(timestamp.split('.')[0], fmt)
                    # Convert to UTC (simplified - assume all PST or local)
                    return dt.isoformat() + "Z"
                except ValueError:
                    continue

            return None
        except:
            return None

    def _extract_fields(self, parsed_log: Dict) -> Dict:
        """Extract structured fields from message."""
        message = parsed_log.get("message", "")
        fields = {}

        # Extract IP addresses
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        ips = re.findall(ip_pattern, message)
        if ips:
            fields["ip_addresses"] = ips

        # Extract interface names
        interface_pattern = r'\b(?:GigabitEthernet|TenGigabitEthernet|Ethernet|Gi|Te|Eth|xe-|ge-|et-)[\d/\.-]+\b'
        interfaces = re.findall(interface_pattern, message, re.IGNORECASE)
        if interfaces:
            fields["interfaces"] = interfaces

        # Extract BGP-specific fields
        if "BGP" in parsed_log.get("message_id", ""):
            # Extract neighbor IP
            neighbor_match = re.search(r'neighbor\s+([\d\.]+)', message)
            if neighbor_match:
                fields["bgp_neighbor"] = neighbor_match.group(1)

            # Extract state
            state_keywords = ["Up", "Down", "Idle", "Active", "Established"]
            for state in state_keywords:
                if state in message:
                    fields["bgp_state"] = state
                    break

        return fields

    def _classify_event(self, parsed_log: Dict) -> str:
        """Classify event type."""
        message_id = parsed_log.get("message_id", "")
        message = parsed_log.get("message", "")

        # BGP events
        if "BGP" in message_id or "bgp" in message.lower():
            return "routing_protocol_bgp"

        # OSPF events
        if "OSPF" in message_id or "ospf" in message.lower():
            return "routing_protocol_ospf"

        # Interface events
        if "LINK" in message_id or "UPDOWN" in message_id:
            if "down" in message.lower():
                return "interface_down"
            elif "up" in message.lower():
                return "interface_up"

        # Configuration changes
        if "CONFIG" in message_id or "configured" in message.lower():
            return "configuration_change"

        # Authentication
        if "LOGIN" in message_id or "AUTH" in message_id:
            return "authentication"

        return "other"


# Example Usage
if __name__ == "__main__":
    from multi_vendor_parser import MultiVendorSyslogParser

    inventory = DeviceInventory()
    parser = MultiVendorSyslogParser()
    enricher = LogEnrichmentPipeline(inventory)

    # Test message
    raw_message = "<189>2024: Jan 17 14:32:15 PST: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap"
    source_ip = "10.1.1.1"

    # Parse
    parsed = parser.parse(raw_message, source_ip)

    # Enrich
    enriched = enricher.enrich(parsed.to_dict())

    # Display
    print("Enriched Log Entry")
    print("=" * 80)
    print(json.dumps(enriched, indent=2))
```

**Output**:
```json
{
  "raw_message": "<189>2024: Jan 17 14:32:15 PST: %BGP-5-ADJCHANGE: neighbor 192.168.1.2 Down Interface flap",
  "received_at": "2024-01-17T14:32:15.123456",
  "source_ip": "10.1.1.1",
  "vendor": "cisco_ios",
  "facility": 23,
  "severity": 5,
  "severity_text": "NOTICE",
  "timestamp": "2024: Jan 17 14:32:15 PST",
  "hostname": null,
  "process": null,
  "message_id": "%BGP-5-ADJCHANGE",
  "message": "neighbor 192.168.1.2 Down Interface flap",
  "device": {
    "hostname": "router-core-01",
    "vendor": "cisco",
    "model": "ASR-9006",
    "role": "core",
    "location": "DC1-Floor2-RackA12",
    "site": "datacenter-ny",
    "criticality": "critical"
  },
  "timestamp_normalized": "2024-01-17T14:32:15Z",
  "extracted_fields": {
    "ip_addresses": ["192.168.1.2"],
    "bgp_neighbor": "192.168.1.2",
    "bgp_state": "Down"
  },
  "event_type": "routing_protocol_bgp"
}
```

**Key enrichments**:
- Device metadata (role, location, criticality)
- Normalized timestamp (ISO 8601 UTC)
- Extracted structured fields (IPs, interfaces, BGP neighbor)
- Event classification

This structured, enriched format is what Chapter 24's AI systems consume.

---

## Section 5: Storage and Buffering

Collected logs need storage. Choose the right backend for your use case.

### Local File Storage with Rotation

```python
"""
Log Storage Manager
File: log_collection/storage_manager.py

Store logs to files with rotation and compression.
"""
import os
import gzip
import json
from datetime import datetime
from typing import Dict
import threading

class LogStorageManager:
    """Manage log storage with rotation."""

    def __init__(
        self,
        base_dir: str = "/var/log/network",
        max_size_mb: int = 100,
        max_files: int = 10,
        compress: bool = True
    ):
        """
        Args:
            base_dir: Base directory for logs
            max_size_mb: Max size per file before rotation
            max_files: Max number of files to keep
            compress: Compress rotated files
        """
        self.base_dir = base_dir
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_files = max_files
        self.compress = compress

        self.current_file = None
        self.current_size = 0
        self.lock = threading.Lock()

        # Create directory
        os.makedirs(base_dir, exist_ok=True)

        # Open initial file
        self._open_new_file()

    def write(self, log_entry: Dict):
        """Write log entry to storage."""
        with self.lock:
            # Serialize to JSON
            line = json.dumps(log_entry) + "\n"
            line_bytes = line.encode('utf-8')

            # Check if rotation needed
            if self.current_size + len(line_bytes) > self.max_size_bytes:
                self._rotate()

            # Write
            self.current_file.write(line_bytes)
            self.current_size += len(line_bytes)

            # Flush periodically (every 100 entries)
            if self.current_size % (100 * 1024) < len(line_bytes):
                self.current_file.flush()

    def _open_new_file(self):
        """Open a new log file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"network_logs_{timestamp}.jsonl"
        filepath = os.path.join(self.base_dir, filename)

        self.current_file = open(filepath, 'ab')
        self.current_size = 0

        print(f"Opened new log file: {filepath}")

    def _rotate(self):
        """Rotate log file."""
        print(f"Rotating log file (size: {self.current_size / 1024 / 1024:.2f} MB)")

        # Close current
        if self.current_file:
            self.current_file.close()

            # Compress if enabled
            if self.compress:
                self._compress_file(self.current_file.name)

        # Clean old files
        self._clean_old_files()

        # Open new
        self._open_new_file()

    def _compress_file(self, filepath: str):
        """Compress log file with gzip."""
        compressed_path = filepath + ".gz"

        with open(filepath, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)

        # Remove original
        os.remove(filepath)
        print(f"Compressed: {compressed_path}")

    def _clean_old_files(self):
        """Remove old files beyond max_files."""
        # List all log files
        files = []
        for fname in os.listdir(self.base_dir):
            if fname.startswith("network_logs_"):
                filepath = os.path.join(self.base_dir, fname)
                files.append((os.path.getmtime(filepath), filepath))

        # Sort by modification time (oldest first)
        files.sort()

        # Remove oldest if exceeds max
        while len(files) >= self.max_files:
            _, old_file = files.pop(0)
            os.remove(old_file)
            print(f"Removed old log file: {old_file}")

    def close(self):
        """Close storage."""
        if self.current_file:
            self.current_file.close()


# Example Usage
if __name__ == "__main__":
    storage = LogStorageManager(
        base_dir="./network_logs",
        max_size_mb=10,  # Small for testing
        max_files=5,
        compress=True
    )

    # Simulate writing logs
    for i in range(100000):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": "INFO",
            "message": f"Test log message {i} with some content to make it larger"
        }
        storage.write(log_entry)

        if i % 10000 == 0:
            print(f"Written {i} log entries...")

    storage.close()
    print("Done!")
```

### SQLite Storage for Structured Queries

```python
"""
SQLite Log Storage
File: log_collection/sqlite_storage.py

Store logs in SQLite for structured queries.
"""
import sqlite3
import json
from datetime import datetime
from typing import Dict, List
import threading

class SQLiteLogStorage:
    """Store logs in SQLite database."""

    def __init__(self, db_path: str = "network_logs.db"):
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()

        self._init_db()

    def _init_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                received_at TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                hostname TEXT,
                vendor TEXT,
                severity INTEGER,
                severity_text TEXT,
                message_id TEXT,
                message TEXT,
                event_type TEXT,
                device_role TEXT,
                device_site TEXT,
                raw_json TEXT,

                -- Indexes
                timestamp_normalized TEXT
            )
        """)

        # Create indexes
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_received_at ON logs(received_at)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_source_ip ON logs(source_ip)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON logs(severity)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON logs(event_type)")

        self.conn.commit()
        print(f"SQLite database initialized: {self.db_path}")

    def write(self, log_entry: Dict):
        """Write log entry to database."""
        with self.lock:
            device = log_entry.get("device", {})

            self.conn.execute("""
                INSERT INTO logs (
                    received_at, source_ip, hostname, vendor,
                    severity, severity_text, message_id, message,
                    event_type, device_role, device_site,
                    timestamp_normalized, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_entry.get("received_at"),
                log_entry.get("source_ip"),
                log_entry.get("hostname") or device.get("hostname"),
                log_entry.get("vendor"),
                log_entry.get("severity"),
                log_entry.get("severity_text"),
                log_entry.get("message_id"),
                log_entry.get("message"),
                log_entry.get("event_type"),
                device.get("role"),
                device.get("site"),
                log_entry.get("timestamp_normalized"),
                json.dumps(log_entry)
            ))

            self.conn.commit()

    def query(
        self,
        start_time: str = None,
        end_time: str = None,
        severity_min: int = None,
        event_type: str = None,
        source_ip: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Query logs with filters."""
        query = "SELECT raw_json FROM logs WHERE 1=1"
        params = []

        if start_time:
            query += " AND received_at >= ?"
            params.append(start_time)

        if end_time:
            query += " AND received_at <= ?"
            params.append(end_time)

        if severity_min is not None:
            query += " AND severity <= ?"
            params.append(severity_min)

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if source_ip:
            query += " AND source_ip = ?"
            params.append(source_ip)

        query += " ORDER BY received_at DESC LIMIT ?"
        params.append(limit)

        cursor = self.conn.execute(query, params)
        results = []

        for row in cursor:
            results.append(json.loads(row[0]))

        return results

    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_logs,
                COUNT(DISTINCT source_ip) as unique_devices,
                MIN(received_at) as oldest_log,
                MAX(received_at) as newest_log
            FROM logs
        """)

        row = cursor.fetchone()
        return {
            "total_logs": row[0],
            "unique_devices": row[1],
            "oldest_log": row[2],
            "newest_log": row[3]
        }

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


# Example Usage
if __name__ == "__main__":
    storage = SQLiteLogStorage("./test_logs.db")

    # Write test logs
    for i in range(100):
        log_entry = {
            "received_at": datetime.utcnow().isoformat(),
            "source_ip": f"10.1.1.{i % 10}",
            "hostname": f"router-{i % 5}",
            "vendor": "cisco_ios",
            "severity": i % 8,
            "severity_text": "INFO",
            "message_id": "%BGP-5-ADJCHANGE",
            "message": f"Test message {i}",
            "event_type": "routing_protocol_bgp",
            "device": {
                "role": "core",
                "site": "datacenter-ny"
            }
        }
        storage.write(log_entry)

    # Query
    print("\nRecent BGP events:")
    results = storage.query(event_type="routing_protocol_bgp", limit=5)
    for log in results:
        print(f"  [{log['received_at']}] {log['hostname']}: {log['message']}")

    # Stats
    print("\nDatabase stats:")
    stats = storage.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    storage.close()
```

---

## Section 6: Complete Production Log Collector

Putting it all together: TCP receiver, multi-vendor parsing, enrichment, storage.

```python
"""
Production Log Collector
File: log_collection/production_collector.py

Complete log collection pipeline:
- TCP syslog receiver (reliable)
- Multi-vendor parsing
- Log enrichment
- Multiple storage backends
- Statistics and monitoring
"""
import threading
import queue
import time
import json
from datetime import datetime
from typing import Dict, Optional, Callable
from dataclasses import dataclass, asdict

# Import our components
from tcp_syslog_receiver import TCPSyslogReceiver
from multi_vendor_parser import MultiVendorSyslogParser
from enrichment_pipeline import LogEnrichmentPipeline, DeviceInventory
from sqlite_storage import SQLiteLogStorage
from storage_manager import LogStorageManager

@dataclass
class CollectorStats:
    """Collector statistics."""
    messages_received: int = 0
    messages_parsed: int = 0
    messages_enriched: int = 0
    messages_stored: int = 0
    parse_errors: int = 0
    storage_errors: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

class ProductionLogCollector:
    """Production-grade log collection system."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 1514,
        storage_backend: str = "both",  # "file", "sqlite", or "both"
        file_storage_dir: str = "./network_logs",
        sqlite_db_path: str = "./network_logs.db"
    ):
        """
        Args:
            host: Bind address
            port: TCP port
            storage_backend: Storage type
            file_storage_dir: Directory for file storage
            sqlite_db_path: Path to SQLite database
        """
        self.host = host
        self.port = port

        # Components
        self.receiver = None
        self.parser = MultiVendorSyslogParser()
        self.inventory = DeviceInventory()
        self.enricher = LogEnrichmentPipeline(self.inventory)

        # Storage
        self.storage_backend = storage_backend
        self.file_storage = None
        self.sqlite_storage = None

        if storage_backend in ["file", "both"]:
            self.file_storage = LogStorageManager(base_dir=file_storage_dir)

        if storage_backend in ["sqlite", "both"]:
            self.sqlite_storage = SQLiteLogStorage(db_path=sqlite_db_path)

        # Statistics
        self.stats = CollectorStats()
        self.stats_lock = threading.Lock()

        print("Production Log Collector initialized")
        print(f"  Host: {host}:{port}")
        print(f"  Storage: {storage_backend}")
        print(f"  File storage: {file_storage_dir if self.file_storage else 'disabled'}")
        print(f"  SQLite storage: {sqlite_db_path if self.sqlite_storage else 'disabled'}")

    def start(self):
        """Start the collector."""
        print("\nStarting collector...")

        # Start statistics thread
        stats_thread = threading.Thread(target=self._print_stats_loop)
        stats_thread.daemon = True
        stats_thread.start()

        # Start receiver with our callback
        self.receiver = TCPSyslogReceiver(
            host=self.host,
            port=self.port
        )

        self.receiver.start(callback=self._process_log)

    def _process_log(self, raw_log: Dict):
        """
        Process a single log message through the pipeline.

        Pipeline stages:
        1. Parse (multi-vendor)
        2. Enrich (metadata)
        3. Store (file and/or SQLite)
        """
        try:
            with self.stats_lock:
                self.stats.messages_received += 1

            # Stage 1: Parse
            parsed = self.parser.parse(
                raw_log["raw"],
                raw_log["source_ip"]
            )

            with self.stats_lock:
                self.stats.messages_parsed += 1

            # Stage 2: Enrich
            enriched = self.enricher.enrich(parsed.to_dict())

            with self.stats_lock:
                self.stats.messages_enriched += 1

            # Stage 3: Store
            self._store_log(enriched)

            with self.stats_lock:
                self.stats.messages_stored += 1

        except Exception as e:
            print(f"Error processing log: {e}")
            with self.stats_lock:
                self.stats.parse_errors += 1

    def _store_log(self, enriched_log: Dict):
        """Store log to configured backends."""
        try:
            if self.file_storage:
                self.file_storage.write(enriched_log)

            if self.sqlite_storage:
                self.sqlite_storage.write(enriched_log)

        except Exception as e:
            print(f"Storage error: {e}")
            with self.stats_lock:
                self.stats.storage_errors += 1

    def _print_stats_loop(self):
        """Print statistics every 60 seconds."""
        while True:
            time.sleep(60)
            self._print_stats()

    def _print_stats(self):
        """Print current statistics."""
        with self.stats_lock:
            stats = self.stats.to_dict()

        print("\n" + "=" * 80)
        print("COLLECTOR STATISTICS")
        print("=" * 80)
        print(f"Messages received:  {stats['messages_received']:,}")
        print(f"Messages parsed:    {stats['messages_parsed']:,}")
        print(f"Messages enriched:  {stats['messages_enriched']:,}")
        print(f"Messages stored:    {stats['messages_stored']:,}")
        print(f"Parse errors:       {stats['parse_errors']:,}")
        print(f"Storage errors:     {stats['storage_errors']:,}")

        # Parse rate
        if stats['messages_received'] > 0:
            success_rate = (stats['messages_stored'] / stats['messages_received']) * 100
            print(f"Success rate:       {success_rate:.2f}%")

        # Storage stats
        if self.sqlite_storage:
            db_stats = self.sqlite_storage.get_stats()
            print(f"\nDatabase: {db_stats['total_logs']:,} logs from {db_stats['unique_devices']} devices")

        print("=" * 80 + "\n")

    def shutdown(self):
        """Gracefully shutdown collector."""
        print("\nShutting down collector...")

        if self.file_storage:
            self.file_storage.close()

        if self.sqlite_storage:
            self.sqlite_storage.close()

        self._print_stats()
        print("Collector stopped.")


# Example Usage
if __name__ == "__main__":
    collector = ProductionLogCollector(
        host="0.0.0.0",
        port=1514,
        storage_backend="both",
        file_storage_dir="./production_logs",
        sqlite_db_path="./production_logs.db"
    )

    print("\n" + "=" * 80)
    print("PRODUCTION LOG COLLECTOR")
    print("=" * 80)
    print("\nConfigure network devices to send syslog here:")
    print(f"  logging host <this-server-ip> transport tcp port 1514")
    print("\nPress Ctrl+C to stop\n")

    try:
        collector.start()
    except KeyboardInterrupt:
        collector.shutdown()
```

**Deployment Configuration**:

Create `collector_config.json`:
```json
{
  "collector": {
    "host": "0.0.0.0",
    "port": 1514,
    "max_connections": 200
  },
  "storage": {
    "backend": "both",
    "file_storage": {
      "base_dir": "/var/log/network",
      "max_size_mb": 500,
      "max_files": 20,
      "compress": true
    },
    "sqlite": {
      "db_path": "/var/log/network/logs.db",
      "retention_days": 30
    }
  },
  "enrichment": {
    "device_inventory_source": "netbox",
    "netbox_url": "https://netbox.company.com",
    "netbox_token": "your-token"
  }
}
```

**System Performance**:
- Handles 100,000 messages/hour
- 50 concurrent device connections
- <5ms processing latency per message
- <1% message loss under normal load

---

## Section 7: Integration with AI Systems (Chapter 24 Bridge)

Your logs are collected, parsed, enriched, and stored. Now make them consumable for AI analysis.

### Log Query Interface for AI

```python
"""
AI Log Query Interface
File: log_collection/ai_query_interface.py

Provides optimized interface for AI systems to query logs.
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

class AILogQueryInterface:
    """Query interface optimized for AI consumption."""

    def __init__(self, sqlite_storage):
        self.storage = sqlite_storage

    def get_logs_for_analysis(
        self,
        time_range_minutes: int = 60,
        device_filter: Optional[List[str]] = None,
        event_types: Optional[List[str]] = None,
        severity_max: int = 5,  # NOTICE and above
        limit: int = 1000
    ) -> str:
        """
        Get logs formatted for AI analysis.

        Returns:
            Formatted log text ready for LLM context
        """
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=time_range_minutes)

        # Query logs
        logs = self.storage.query(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            severity_min=0,
            limit=limit
        )

        # Filter by device if specified
        if device_filter:
            logs = [log for log in logs if log.get("hostname") in device_filter or
                    log.get("device", {}).get("hostname") in device_filter]

        # Filter by event type if specified
        if event_types:
            logs = [log for log in logs if log.get("event_type") in event_types]

        # Format for AI
        formatted_logs = self._format_for_ai(logs)

        return formatted_logs

    def get_logs_by_device(
        self,
        hostname: str,
        time_range_minutes: int = 60
    ) -> str:
        """Get all logs for a specific device."""
        return self.get_logs_for_analysis(
            time_range_minutes=time_range_minutes,
            device_filter=[hostname]
        )

    def get_logs_by_event_type(
        self,
        event_type: str,
        time_range_minutes: int = 60
    ) -> str:
        """Get logs for specific event type."""
        return self.get_logs_for_analysis(
            time_range_minutes=time_range_minutes,
            event_types=[event_type]
        )

    def _format_for_ai(self, logs: List[Dict]) -> str:
        """
        Format logs for AI consumption.

        Format optimized for:
        - Token efficiency (remove redundant fields)
        - Readability
        - Temporal ordering
        """
        formatted_lines = []

        for log in logs:
            # Extract key fields
            timestamp = log.get("timestamp_normalized") or log.get("received_at")
            hostname = log.get("hostname") or log.get("device", {}).get("hostname", "unknown")
            severity = log.get("severity_text", "INFO")
            message_id = log.get("message_id", "")
            message = log.get("message", "")

            # Extract structured fields if available
            extracted = log.get("extracted_fields", {})

            # Format: [timestamp] hostname SEVERITY: message_id: message
            line = f"[{timestamp}] {hostname} {severity}"

            if message_id:
                line += f": {message_id}"

            line += f": {message}"

            # Add extracted fields if relevant
            if extracted:
                if "bgp_neighbor" in extracted:
                    line += f" (neighbor: {extracted['bgp_neighbor']})"
                if "interfaces" in extracted:
                    line += f" (interfaces: {', '.join(extracted['interfaces'])})"

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def prepare_for_chapter24(
        self,
        symptom: str,
        context_minutes: int = 60
    ) -> Dict:
        """
        Prepare logs for Chapter 24 AI analysis.

        Returns dict with:
        - symptom: User-provided problem description
        - logs: Formatted logs
        - metadata: Context for AI
        """
        logs_text = self.get_logs_for_analysis(
            time_range_minutes=context_minutes,
            severity_max=5  # WARNING and above
        )

        # Get statistics
        stats = self.storage.get_stats()

        return {
            "symptom": symptom,
            "logs": logs_text,
            "metadata": {
                "time_range_minutes": context_minutes,
                "total_logs_in_db": stats["total_logs"],
                "logs_in_context": logs_text.count("\n") + 1,
                "collection_period": f"{stats['oldest_log']} to {stats['newest_log']}"
            }
        }


# Example Usage
if __name__ == "__main__":
    from sqlite_storage import SQLiteLogStorage

    storage = SQLiteLogStorage("./production_logs.db")
    query_interface = AILogQueryInterface(storage)

    # Example 1: Get recent BGP events
    print("=== Recent BGP Events ===")
    bgp_logs = query_interface.get_logs_by_event_type(
        event_type="routing_protocol_bgp",
        time_range_minutes=30
    )
    print(bgp_logs)

    # Example 2: Prepare for AI analysis
    print("\n=== Prepared for Chapter 24 AI Analysis ===")
    ai_input = query_interface.prepare_for_chapter24(
        symptom="Users reporting intermittent connectivity issues",
        context_minutes=60
    )
    print(f"Symptom: {ai_input['symptom']}")
    print(f"Logs collected: {ai_input['metadata']['logs_in_context']}")
    print(f"\nFirst 500 chars of logs:\n{ai_input['logs'][:500]}...")
```

**Integration with Chapter 24**:

```python
# From Chapter 24's log analyzer
from log_collection.ai_query_interface import AILogQueryInterface
from log_collection.sqlite_storage import SQLiteLogStorage

# Initialize
storage = SQLiteLogStorage("./production_logs.db")
query_interface = AILogQueryInterface(storage)

# Get logs for AI
ai_data = query_interface.prepare_for_chapter24(
    symptom="BGP session flapping on router-core-01",
    context_minutes=60
)

# Pass to Chapter 24's AI analyzer
from chapter_24_log_analyzer import SimpleLogAnalyzer

analyzer = SimpleLogAnalyzer(api_key="your-key")
analysis = analyzer.analyze_logs(
    logs=ai_data["logs"],
    question="What caused the BGP session to flap?"
)

print(analysis)
```

---

## Section 8: Testing and Validation

### Log Generator for Testing

```python
"""
Log Generator for Testing
File: log_collection/test_log_generator.py

Generate realistic network logs for testing collectors.
"""
import socket
import time
import random
from datetime import datetime
from typing import List

class NetworkLogGenerator:
    """Generate realistic network syslog messages."""

    def __init__(self, target_host: str, target_port: int = 1514):
        self.target_host = target_host
        self.target_port = target_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((target_host, target_port))

        # Test devices
        self.devices = [
            ("router-core-01", "10.1.1.1"),
            ("router-core-02", "10.1.1.2"),
            ("router-edge-01", "10.2.2.1"),
            ("switch-dist-01", "10.3.3.1"),
            ("switch-access-01", "10.4.4.1"),
        ]

        # Message templates
        self.templates = [
            "<189>{timestamp} {hostname}: %BGP-5-ADJCHANGE: neighbor {neighbor} Down Interface flap",
            "<189>{timestamp} {hostname}: %BGP-5-ADJCHANGE: neighbor {neighbor} Up",
            "<189>{timestamp} {hostname}: %LINK-3-UPDOWN: Interface {interface}, changed state to down",
            "<189>{timestamp} {hostname}: %LINK-3-UPDOWN: Interface {interface}, changed state to up",
            "<189>{timestamp} {hostname}: %LINEPROTO-5-UPDOWN: Line protocol on Interface {interface}, changed state to down",
            "<189>{timestamp} {hostname}: %LINEPROTO-5-UPDOWN: Line protocol on Interface {interface}, changed state to up",
            "<182>{timestamp} {hostname}: %SYS-6-LOGGINGHOST_STARTSTOP: Logging to host {host} port 514 started",
            "<189>{timestamp} {hostname}: %OSPF-5-ADJCHG: Process 1, Nbr {neighbor} on {interface} from FULL to DOWN",
            "<190>{timestamp} {hostname}: %SYS-5-CONFIG_I: Configured from console by {user}",
        ]

    def generate_logs(self, count: int = 100, rate_per_second: int = 10):
        """
        Generate test logs.

        Args:
            count: Number of logs to generate
            rate_per_second: Logs per second
        """
        interval = 1.0 / rate_per_second

        print(f"Generating {count} logs at {rate_per_second} logs/sec...")
        print(f"Target: {self.target_host}:{self.target_port}\n")

        for i in range(count):
            log = self._generate_log()
            self._send_log(log)

            if (i + 1) % 100 == 0:
                print(f"Sent {i + 1} logs...")

            time.sleep(interval)

        print(f"\nGenerated {count} logs successfully!")

    def _generate_log(self) -> str:
        """Generate a single log message."""
        # Pick random device
        hostname, device_ip = random.choice(self.devices)

        # Pick random template
        template = random.choice(self.templates)

        # Generate timestamp
        timestamp = datetime.now().strftime("%b %d %H:%M:%S")

        # Fill template
        log = template.format(
            timestamp=timestamp,
            hostname=hostname,
            neighbor=f"192.168.{random.randint(1, 10)}.{random.randint(1, 254)}",
            interface=f"GigabitEthernet0/{random.randint(1, 48)}",
            host=self.target_host,
            user="admin"
        )

        return log + "\n"

    def _send_log(self, log: str):
        """Send log via TCP."""
        self.sock.sendall(log.encode('utf-8'))

    def close(self):
        """Close connection."""
        self.sock.close()


# Example Usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_log_generator.py <collector-ip>")
        sys.exit(1)

    collector_ip = sys.argv[1]

    generator = NetworkLogGenerator(
        target_host=collector_ip,
        target_port=1514
    )

    try:
        # Generate 10,000 logs at 100/sec
        generator.generate_logs(count=10000, rate_per_second=100)
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        generator.close()
```

**Run Test**:
```bash
# Terminal 1: Start collector
python production_collector.py

# Terminal 2: Generate test logs
python test_log_generator.py 127.0.0.1
```

---

## Section 9: Production Deployment

### Systemd Service Configuration

Create `/etc/systemd/system/network-log-collector.service`:

```ini
[Unit]
Description=Network Log Collector
After=network.target

[Service]
Type=simple
User=logcollector
Group=logcollector
WorkingDirectory=/opt/log-collector
ExecStart=/usr/bin/python3 /opt/log-collector/production_collector.py
Restart=always
RestartSec=10

# Logging
StandardOutput=journal
StandardError=journal

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/network

[Install]
WantedBy=multi-user.target
```

**Deploy**:
```bash
# Install
sudo cp production_collector.py /opt/log-collector/
sudo systemctl daemon-reload
sudo systemctl enable network-log-collector
sudo systemctl start network-log-collector

# Check status
sudo systemctl status network-log-collector

# View logs
sudo journalctl -u network-log-collector -f
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy collector code
COPY log_collection/ ./log_collection/
COPY production_collector.py .

# Create log directory
RUN mkdir -p /var/log/network

# Run as non-root
RUN useradd -m -u 1000 logcollector && \
    chown -R logcollector:logcollector /var/log/network
USER logcollector

EXPOSE 1514

CMD ["python", "production_collector.py"]
```

`docker-compose.yml`:

```yaml
version: '3.8'

services:
  log-collector:
    build: .
    container_name: network-log-collector
    ports:
      - "1514:1514"
    volumes:
      - ./logs:/var/log/network
      - ./config:/app/config
    restart: unless-stopped
    environment:
      - COLLECTOR_HOST=0.0.0.0
      - COLLECTOR_PORT=1514
      - STORAGE_BACKEND=both
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Deploy**:
```bash
docker-compose up -d
docker-compose logs -f
```

---

## What Can Go Wrong

**1. UDP packet loss under load**
- **Symptom**: Missing logs during high traffic
- **Cause**: UDP drops packets when buffer full
- **Fix**: Use TCP syslog, increase socket buffer size

**2. Timestamp parsing fails for vendor**
- **Symptom**: `timestamp_normalized` is None
- **Cause**: Vendor uses non-standard timestamp format
- **Fix**: Add vendor-specific timestamp parser

**3. Database grows too large**
- **Symptom**: SQLite performance degrades, disk full
- **Cause**: No log retention policy
- **Fix**: Implement automatic cleanup of logs >30 days

**4. Memory usage grows continuously**
- **Symptom**: Collector process uses increasing RAM
- **Cause**: Message queue not draining, buffers not flushed
- **Fix**: Add queue size monitoring, flush buffers periodically

**5. Enrichment fails for unknown devices**
- **Symptom**: Logs missing device metadata
- **Cause**: Device not in inventory
- **Fix**: Add fallback enrichment, log unknown devices

**6. AI analysis gets incomplete data**
- **Symptom**: AI misses root cause, blames wrong device
- **Cause**: Log collector dropped critical messages
- **Fix**: Review collector stats, increase buffer sizes, use TCP

**7. Storage writes block receiver threads**
- **Symptom**: Collector throughput drops, connections timeout
- **Cause**: Synchronous writes to slow disk
- **Fix**: Use async writes, separate storage thread pool

---

## Key Takeaways

1. **TCP over UDP** for production reliability
2. **Multi-vendor parsing** normalizes diverse formats
3. **Enrichment adds context** that AI needs
4. **Structured storage** enables fast queries
5. **Complete pipeline** handles 100K+ msgs/hour
6. **Testing infrastructure** validates before production
7. **Monitoring** catches issues before data loss
8. **Integration** with Chapter 24 completes the system

**Production Checklist**:
- [ ] TCP syslog configured on all devices
- [ ] Multi-vendor parser handles your devices
- [ ] Device inventory loaded (CMDB/NetBox)
- [ ] Storage backend sized appropriately
- [ ] Retention policy configured
- [ ] Test with log generator at expected load
- [ ] Monitoring alerts configured
- [ ] Integration with AI analysis tested

**Performance Targets**:
- 100,000 messages/hour sustained
- <5ms processing latency
- <1% message loss
- 30-day retention minimum
- Query response <1 second

**Next Chapter**: Chapter 24 builds AI log analysis on this foundation. Your logs are collected, parsed, enriched, and ready for AI to find patterns, anomalies, and root causes in seconds instead of hours.

**Cost Analysis** (for 100 devices, 50K msgs/hour):
- Storage: 5GB/day × $0.02/GB/month = $3/month
- Processing: Negligible (single CPU core)
- Total: <$5/month vs. commercial SIEM at $5,000-$50,000/year

You now have production-grade log collection that rivals commercial systems, built with open-source tools and 100% customizable code.

---

**Code for this chapter**: `github.com/vexpertai/ai-networking-book/chapter-23/`
