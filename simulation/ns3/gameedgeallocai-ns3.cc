/*
 * Offline NS-3 replay of the GameEdgeAllocAI Python allocation trace.
 * Compatible target: ns-3 3.45+ with wifi, internet, applications, flow-monitor.
 * Compile as scratch/gameedgeallocai-ns3.cc using ./ns3 run.
 *
 * The input bandwidth is implemented as APPLICATION-LEVEL offered-rate shaping,
 * not as guaranteed PHY airtime reservation or a coupled Stackelberg controller.
 * FlowMonitor provides measured wireless packet delivery and packet delay.
 * Python retains responsibility for the CPU/compute-time allocation.
 */
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cmath>
#include <stdexcept>
using namespace ns3;

struct Row {
  uint32_t slot{0};
  uint32_t user{0};
  double dataMbit{0};
  double rateMHz{0};
  double spectralEff{1};
};

static std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);std::string v;
  while(std::getline(ss,v,','))out.push_back(v);
  return out;
}

static std::vector<Row> ReadTrace(const std::string& path) {
  std::ifstream file(path);
  if(!file)throw std::runtime_error("Cannot open allocation trace: "+path);
  std::string line;
  if(!std::getline(file,line))throw std::runtime_error("Empty allocation trace");
  std::vector<std::string> names=Split(line);
  std::map<std::string,size_t> fields;
  for(size_t i=0;i<names.size();++i)fields[names[i]]=i;
  for(const auto& name: {"slot","user_id","data_mbit","bandwidth_mhz","spectral_eff"})
    if(fields.count(name)==0)throw std::runtime_error(std::string("Missing CSV field: ")+name);
  std::vector<Row> rows;
  while(std::getline(file,line)){
    if(line.empty())continue;
    auto v=Split(line);
    if(v.size()<names.size())continue;
    Row r;
    r.slot=static_cast<uint32_t>(std::stoul(v.at(fields.at("slot"))));
    r.user=static_cast<uint32_t>(std::stoul(v.at(fields.at("user_id"))));
    r.dataMbit=std::stod(v.at(fields.at("data_mbit")));
    r.rateMHz=std::stod(v.at(fields.at("bandwidth_mhz")));
    r.spectralEff=std::stod(v.at(fields.at("spectral_eff")));
    if(r.dataMbit>0 && r.rateMHz>0)rows.push_back(r);
  }
  return rows;
}

int main(int argc,char* argv[]){
  std::string input="allocations_gameedgealloc.csv";
  std::string output="ns3_flow_metrics.csv";
  double slotSeconds=1.0;
  uint32_t seed=42;
  CommandLine cmd(__FILE__);
  cmd.AddValue("input","Full path to Python allocation trace",input);
  cmd.AddValue("output","Output packet-flow CSV path",output);
  cmd.AddValue("slotSeconds","Duration of one workload slot in seconds",slotSeconds);
  cmd.AddValue("seed","Random seed",seed);
  cmd.Parse(argc,argv);
  if(slotSeconds<=0)throw std::runtime_error("slotSeconds must be positive");
  auto rows=ReadTrace(input);
  if(rows.empty())throw std::runtime_error("Trace has no positive allocation rows");
  uint32_t users=0,maxSlot=0;
  for(const auto& r:rows){users=std::max(users,r.user+1);maxSlot=std::max(maxSlot,r.slot);}
  if(users>1000)throw std::runtime_error("Too many users for this access-point example");
  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(1);

  NodeContainer stations;stations.Create(users);
  NodeContainer ap;ap.Create(1);
  WifiHelper wifi;
  wifi.SetStandard(WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager("ns3::IdealWifiManager");
  YansWifiChannelHelper channel=YansWifiChannelHelper::Default();
  YansWifiPhyHelper phy;
  phy.SetChannel(channel.Create());
  WifiMacHelper mac;
  Ssid ssid=Ssid("GameEdgeAllocAI");
  mac.SetType("ns3::StaWifiMac","Ssid",SsidValue(ssid),
              "ActiveProbing",BooleanValue(false));
  NetDeviceContainer staDevices=wifi.Install(phy,mac,stations);
  mac.SetType("ns3::ApWifiMac","Ssid",SsidValue(ssid));
  NetDeviceContainer apDevices=wifi.Install(phy,mac,ap);

  MobilityHelper staMobility;
  Ptr<ListPositionAllocator> positions=CreateObject<ListPositionAllocator>();
  for(uint32_t i=0;i<users;i++){
    const double angle=6.283185307179586*static_cast<double>(i)/users;
    const double radius=8.0+static_cast<double>(i%7)*2.0;
    positions->Add(Vector(radius*std::cos(angle),radius*std::sin(angle),0.0));
  }
  staMobility.SetPositionAllocator(positions);
  staMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  staMobility.Install(stations);
  MobilityHelper apMobility;
  apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  Ptr<ListPositionAllocator> apPos=CreateObject<ListPositionAllocator>();
  apPos->Add(Vector(0,0,0));apMobility.SetPositionAllocator(apPos);apMobility.Install(ap);

  InternetStackHelper stack;stack.Install(stations);stack.Install(ap);
  Ipv4AddressHelper addresses;addresses.SetBase("10.17.0.0","255.255.0.0");
  addresses.Assign(staDevices);
  Ipv4InterfaceContainer apInterfaces=addresses.Assign(apDevices);
  Ipv4Address server=apInterfaces.GetAddress(0);
  const double startOffset=2.0; // allow Wi-Fi association before first payload
  for(uint32_t i=0;i<users;++i){
    const uint16_t port=static_cast<uint16_t>(9000+i);
    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(),port));
    auto apps=sink.Install(ap.Get(0));
    apps.Start(Seconds(0.0));
    apps.Stop(Seconds(startOffset+(maxSlot+1)*slotSeconds+1.0));
  }
  for(const auto& r:rows){
    const uint16_t port=static_cast<uint16_t>(9000+r.user);
    OnOffHelper app("ns3::UdpSocketFactory",InetSocketAddress(server,port));
    // Soft offered-rate limit: bandwidth_mhz * spectral_eff -> megabit/second.
    const double offeredMbitPerSecond=std::max(0.001,r.rateMHz*r.spectralEff);
    const uint64_t bitsPerSecond=static_cast<uint64_t>(offeredMbitPerSecond*1e6);
    app.SetAttribute("DataRate",DataRateValue(DataRate(bitsPerSecond)));
    app.SetAttribute("PacketSize",UintegerValue(1000));
    app.SetAttribute("MaxBytes",UintegerValue(static_cast<uint64_t>(r.dataMbit*1e6/8.0)));
    app.SetAttribute("OnTime",StringValue("ns3::ConstantRandomVariable[Constant=1]"));
    app.SetAttribute("OffTime",StringValue("ns3::ConstantRandomVariable[Constant=0]"));
    const double start=startOffset+r.slot*slotSeconds;
    auto apps=app.Install(stations.Get(r.user));
    apps.Start(Seconds(start+0.002));
    apps.Stop(Seconds(start+slotSeconds-0.002));
  }
  FlowMonitorHelper fm;
  Ptr<FlowMonitor> monitor=fm.InstallAll();
  const double end=startOffset+(maxSlot+1)*slotSeconds+1.0;
  Simulator::Stop(Seconds(end));
  Simulator::Run();
  monitor->CheckForLostPackets();
  Ptr<Ipv4FlowClassifier> classifier=DynamicCast<Ipv4FlowClassifier>(fm.GetClassifier());
  struct FlowTotals {uint64_t tx{0},rx{0},lost{0},bytes{0};double delay{0};};
  std::vector<FlowTotals> totals(users);
  for(const auto& entry:monitor->GetFlowStats()){
    auto tuple=classifier->FindFlow(entry.first);
    if(tuple.destinationAddress!=server)continue;
    if(tuple.destinationPort<9000 || tuple.destinationPort>=9000+users)continue;
    uint32_t i=tuple.destinationPort-9000;
    const auto& s=entry.second;
    totals[i].tx+=s.txPackets;totals[i].rx+=s.rxPackets;
    totals[i].lost+=s.lostPackets;totals[i].bytes+=s.rxBytes;
    totals[i].delay+=s.delaySum.GetSeconds();
  }
  std::ofstream h(output);
  if(!h)throw std::runtime_error("Cannot write flow summary: "+output);
  h<<"user_id,tx_packets,rx_packets,rx_bytes,lost_packets,mean_delay_s,throughput_mbps\n";
  h<<std::setprecision(12);
  for(uint32_t i=0;i<users;i++){
    const auto& s=totals[i];
    const double avgDelay=s.rx? s.delay/s.rx:0.0;
    const double mbps=s.bytes*8.0/((maxSlot+1)*slotSeconds*1e6);
    h<<i<<','<<s.tx<<','<<s.rx<<','<<s.bytes<<','<<s.lost<<','<<avgDelay<<','<<mbps<<'\n';
  }
  h.close();Simulator::Destroy();
  return 0;
}
