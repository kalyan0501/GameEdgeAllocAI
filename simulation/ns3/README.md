# NS-3 packet-level replay (optional dependency)

`gameedgeallocai-ns3.cc` creates an infrastructure Wi-Fi BSS (one AP, a dynamic number of stations), sends the workload described by a Python-generated `allocations_*.csv`, and records **actual packet-level** deliveries and mean FlowMonitor packet delays. The code targets ns-3 **3.45+** with Wi-Fi, applications, internet and flow-monitor libraries.

```bash
# In project root: generate a short trace first
python -m gameedgealloc.cli demo --users 8 --slots 12 --output results/demo
# With an existing ns-3 checkout and build:
bash simulation/ns3/run_ns3.sh /absolute/path/to/ns-3 \
  "$PWD/results/demo/allocations_gameedgealloc.csv" \
  "$PWD/results/demo/ns3_flow_metrics.csv"
python -m gameedgealloc.cli ns3-report \
  --input results/demo/ns3_flow_metrics.csv \
  --output results/demo/ns3_report.csv
```

This is a working **trace-driven packet simulation**, not an online Python↔NS-3 live closed-loop interface. `bandwidth_mhz × spectral_eff` controls an application's *offered UDP bit rate*, but does **not** guarantee PHY/MAC bandwidth reservation. CPU resource decisions remain Python-side. Do not interpret FlowMonitor's mean packet delay as whole-task completion latency. TCP and edge CPU execution are not modelled by this example. The first two seconds allow Wi-Fi association.

The package can run fully in Python without ns-3. Since ns-3 is not installed in the packaging environment, the optional C++ module is provided as source and cannot be claimed to have been compiled or run there; its syntax follows the official ns-3 tutorial patterns.
