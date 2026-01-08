using TOML
using AdaptiveGorilla


file = "$(@__DIR__)/models/ta.toml"

DATASET = "most"
DPATH   = "/spaths/datasets/$(DATASET)/dataset.json"
WM = InertiaWM(;
               object_rate = 8.0,
               area_width = 720.0,
               area_height = 480.0,
               birth_weight = 0.01,
               single_size = 5.0,
               single_noise = 0.15,
               single_cpoisson_log_penalty = 1.1,
               stability = 0.75,
               vel = 4.5,
               force_low = 3.0,
               force_high = 10.0,
               material_noise = 0.01,
               ensemble_var_shift = 0.1)
SCENE = 1
color = Light
FRAMES = 120
SHOW_GORILLA = true

experiment = MostExp(DPATH, WM, SCENE, color, FRAMES, SHOW_GORILLA)
agent = load_agent(file, experiment.init_query);
