#include "isaaclab/algorithms/recurrent_height_map_runner.h"
#include "utils/PolicyHeightMap.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include <unitree/robot/channel/channel_publisher.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <thread>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

class StationaryRobot : public isaaclab::Articulation {
public:
    void update() override {
        data.joint_pos=data.default_joint_pos;
        data.joint_vel.setZero();data.root_ang_vel_b.setZero();
        data.projected_gravity_b=Eigen::Vector3f(0,0,-1);
        data.root_quat_w=Eigen::Quaternionf::Identity();
    }
};

int main(int argc, char** argv)
{
    try {
        require(argc == 4, "Usage: recurrent_height_map_test MODEL REFERENCE_JSON DEPLOY_YAML");
        std::ifstream file(argv[2]);
        const auto fixture = nlohmann::json::parse(file);
        std::vector<float> image;
        isaaclab::RecurrentHeightMapRunner runner(argv[1], [&] { return image; });
        float max_error = 0;
        for (int repeat = 0; repeat < 2; ++repeat) {
            runner.reset();
            for (const auto& frame : fixture) {
                image = frame.at("depth").get<std::vector<float>>();
                auto result = runner.forward({{"single_obs",frame.at("single_obs").get<std::vector<float>>()}});
                for (const auto* name : {"actions","obs_history_out","h_out","c_out"}) {
                    const auto expected = frame.at(name).get<std::vector<float>>();
                    require(result.at(name).size() == expected.size(), "Output shape mismatch");
                    for (std::size_t i=0;i<expected.size();++i) {
                        require(std::isfinite(result.at(name)[i]), "Nonfinite recurrent output");
                        max_error = std::max(max_error,std::abs(expected[i]-result.at(name)[i]));
                    }
                }
            }
        }
        require(max_error < 1e-4f,"C++/Python sequence or reset mismatch");
        bool rejected = false;
        try { runner.forward({{"single_obs",std::vector<float>(97)}}); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected,"Wrong input size must be rejected before ORT reads memory");

        // Isolated loopback domain: no motor publisher or robot connection.
        unitree::robot::ChannelFactory::Instance()->Init(87,"lo");
        using Message = unitree_go::msg::dds_::HeightMap_;
        utils::PolicyHeightMap map;
        map.start("rt/g1/test_policy_height_map",.2);
        unitree::robot::ChannelPublisher<Message> heights("rt/g1/test_policy_height_map");
        unitree::robot::ChannelPublisher<Message> masks("rt/g1/test_policy_height_map_missing_mask");
        heights.InitChannel(); masks.InitChannel();
        std::vector<float> sample;
        require(!map.sample(sample) && sample[187]==1,"Missing input is unknown");
        Message h;
        h.frame_id()="base_yaw_relative"; h.width()=17; h.height()=11;
        h.resolution()=.1f; h.origin()={-.8f,-.5f}; h.data().resize(187);
        for(int y=0;y<11;++y) for(int x=0;x<17;++x) h.data()[y*17+x]=x*.03f+y*.001f;
        auto mask=h; mask.data().assign(187,0); mask.data()[3*17+7]=1;
        bool ready=false;
        for(int attempt=0;attempt<100 && !ready;++attempt) {
            h.stamp()=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            mask.stamp()=h.stamp(); heights.Write(h); masks.Write(mask);
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); ready=map.sample(sample);
        }
        require(ready,"Paired DDS map not received");
        require(std::abs(sample[8*11+4]-.244f)<1e-6f,"DDS y-major to policy x-major transpose");
        require(sample[7*11+3]==0 && sample[187+7*11+3]==1,"Missing cells must zero elevation");
        // An unmatched elevation must not replace a complete pair.
        h.stamp()+=.001; h.data().assign(187,1); heights.Write(h);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        require(map.sample(sample) && std::abs(sample[8*11+4]-.244f)<1e-6f,"Unmatched stamps consumed");
        std::this_thread::sleep_for(std::chrono::milliseconds(220));
        require(!map.sample(sample) && sample[187]==1,"Stale map must be unknown");
        auto config=YAML::LoadFile(argv[3]);
        config["height_map"]["topic"]="rt/g1/test_policy_height_map";
        config["height_map"]["timeout_s"]=.2;
        auto robot=std::make_shared<StationaryRobot>();
        unitree::common::UnitreeJoystick joystick;
        robot->data.joystick=&joystick;
        isaaclab::ManagerBasedRLEnv env(config,robot);
        env.alg=std::make_unique<isaaclab::RecurrentHeightMapRunner>(argv[1],[&]{return env.height_map_image;});
        env.fixed_command_enabled=env.fixed_command_active=true;env.fixed_lin_vel_x=.2f;
        env.reset();env.step();
        auto observations=env.observation_manager->last_group_observations().at(0).second;
        require(observations.size()==98,"Deployment must provide one frame, not 980 history values");
        require(observations[6]==0,"Missing map must stop commanded velocity");
        require(env.action_manager->processed_actions().size()==29,"Deployment action dimension");
        for(int attempt=0;attempt<100 && !env.height_map_ready;++attempt) {
            h.stamp()=std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
            mask.stamp()=h.stamp();heights.Write(h);masks.Write(mask);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));env.step();
        }
        require(env.height_map_ready,"Configured environment failed to receive map");
        observations=env.observation_manager->last_group_observations().at(0).second;
        require(std::abs(observations[6]-.2f)<1e-6f,"Fresh map must allow configured velocity");
        require(std::abs(observations[5]+1)<1e-6f,"Gravity observation layout");
        for(float q:env.action_manager->processed_actions()) require(std::isfinite(q),"Nonfinite target angle");
        std::cout << "Recurrent sequence/reset parity max error " << max_error << "; DDS pairing/transpose/expiry passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
