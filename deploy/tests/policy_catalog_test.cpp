#include "utils/PolicyCatalog.h"
#include "utils/ExternalVelocity.h"
#include <cassert>
#include <iostream>

int main(int argc, char** argv) {
    using namespace robot_automation;
    if (argc == 3 && std::string(argv[1]) == "--go2-config") {
        const auto fsm = YAML::LoadFile(argv[2])["FSM"];
        nlohmann::json policies = nlohmann::json::object();
        for (const auto* name : {"Velocity_Up", "Velocity_Down", "Velocity_Left", "Velocity_Right"}) {
            assert(fsm["_"][name]);
            assert(ExternalVelocity::accepts_external_velocity(name));
            policies[name] = "/models/" + std::string(name) + "/resolved-run";
        }
        const auto catalog = make_policy_catalog(fsm, policies);
        assert(catalog["states"].size() == fsm["_"].size());
        assert(catalog["overlays"].empty());
        for (const auto& state : catalog["states"]) {
            const auto name = state["name"].get<std::string>();
            if (!policies.contains(name)) continue;
            assert(state["policy_name"] == "resolved-run");
            assert(state["configured_policy_dir"] == fsm[name]["policy_dir"].as<std::string>());
            assert(state["policy_dir"] == policies[name]);
            assert(state["enabled"] == true);
            bool from_stand = false;
            for (const auto& source : catalog["states"])
                if (source["name"] == "FixStand")
                    for (const auto& transition : source["transitions"])
                        if (transition["target"] == name) from_stand = true;
            assert(from_stand);
        }
        std::cout << "Go2 four-direction policy bindings passed\n";
        return 0;
    }
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
    assert(bridge.handle({{"type", "policy_catalog"}})["ok"] == false);
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
    const auto previous_id = after["policy_catalog_id"];
    bridge.set_policy_catalog(catalog);
    const auto refreshed = bridge.handle({{"type", "policy_catalog"}});
    assert(refreshed["policy_catalog"]["id"] != previous_id);
    assert(bridge.handle(request("status"))["armed"] == true);
    std::cout << "policy catalog and read-only lease isolation passed\n";
}
