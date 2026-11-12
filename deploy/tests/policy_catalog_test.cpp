#include "utils/PolicyCatalog.h"
#include "utils/ExternalVelocity.h"
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    using namespace robot_automation;
    auto cfg = YAML::Load(R"(
_: {Passive: {id: 1}, Walk: {id: 2, type: RLBase}, Missing: {id: 3}}
Passive:
  transitions: {Walk: 'RB + X.on_pressed', NotRegistered: 'RT + X.on_pressed'}
Walk:
  policy_dir: /models/parent
  transitions: {Passive: 'LT + B.on_pressed', Missing: 'RB + Y.on_pressed'}
Missing:
  policy_dir: /models/missing
)");
    auto catalog = make_policy_catalog(cfg, {{"Walk", "/models/parent/actual-run"}},
        {{"Missing", "model.onnx not found"}},
        {{{"target", "Walk"}, {"expression", "RT + A.on_pressed"}}});
    assert(catalog["states"].size() == 3);
    assert(catalog["states"][0]["transitions"].size() == 1);
    assert(catalog["states"][1]["policy_name"] == "actual-run");
    assert(catalog["states"][1]["configured_policy_dir"] == "/models/parent");
    assert(catalog["states"][2]["enabled"] == false);
    assert(catalog["states"][2]["reason"] == "model.onnx not found");
    assert(catalog["states"][2]["policy_dir"] == "");
    assert(catalog["overlays"].size() == 1);
    auto& bridge = ExternalVelocity::instance();
    bridge.record_policy("Walk", "/models/parent/actual-run");
    assert(bridge.policy_directories()["Walk"] == "/models/parent/actual-run");
    bridge.record_overlay({{"target", "Walk"}, {"expression", "RT + A.on_pressed"}});
    assert(bridge.policy_overlays().size() == 1);
    bridge.set_policy_catalog(catalog);
    if (argc == 2) {
        bridge.start(argv[1]);
        std::cout << bridge.handle({{"type", "status"}})["policy_catalog_id"].get<std::string>() << std::endl;
        std::string stop; std::getline(std::cin, stop);
        return 0;
    }
    const std::string owner(32, 'a');
    const auto state = ExternalVelocity::accepts_external_velocity("Velocity_X") ? "Velocity_X" : "Velocity_Up";
    bridge.update(state, false, true);
    assert(bridge.set_ranges({{{-1,1},{-1,1},{-1,1}}}, false));
    auto request = [&](const char* type) { return nlohmann::json{
        {"type", type}, {"session", owner}, {"request_id", "test"}, {"sent_mono_s", ExternalVelocity::now()}}; };
    assert(bridge.handle(request("arm"))["ok"] == true);
    auto before = bridge.handle(request("status"));
    auto query = bridge.handle({{"type", "policy_catalog"}, {"request_id", "read-only"}});
    auto after = bridge.handle(request("status"));
    assert(query["ok"] == true && query["request_id"] == "read-only");
    assert(query["policy_catalog"]["id"] == after["policy_catalog_id"]);
    assert(!after.contains("policy_catalog"));
    for (const auto* key : {"state", "armed", "owns_lease", "generation", "velocity", "remote_active"})
        assert(before[key] == after[key]);
    assert(after["armed"] == true);
    std::cout << "policy catalog and read-only lease isolation passed\n";
}
