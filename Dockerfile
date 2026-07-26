# Serves the Level 02 + Level 03 webapp (webapp/app.py) headlessly.
#
# Build:  docker build -t g1-webapp .
# Run:    docker run -p 8000:8000 g1-webapp
# Then open http://localhost:8000
#
# MUJOCO_GL=osmesa below renders on CPU with no GPU/driver dependency --
# works on any plain VM/container host. If you deploy to a GPU instance
# instead, switch it to MUJOCO_GL=egl (same code, no other changes).

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglfw3 \
    libglew-dev \
    && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY webapp/requirements-web.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-web.txt

# --- App code ---
COPY webapp/ webapp/
COPY envs/ envs/
COPY assets/ assets/
COPY scripts/ scripts/

# --- Level 03 RL code: only the files webapp/runner_level03.py actually
# imports, not training_extension's ~30MB of unrelated training artifacts
# (rl_artifacts/, cem_artifacts/milestones/, checkpoint sweeps, etc.) ---
COPY training_extension/__init__.py training_extension/optimize_direct.py \
     training_extension/sac_parameter_env.py training_extension/basketball_env.py \
     training_extension/derived_baseline.py training_extension/td3_residual_env.py \
     training_extension/scene_throw_LEVEL03_ring.xml \
     training_extension/
COPY training_extension/cem_artifacts/selected/state.json \
     training_extension/cem_artifacts/selected/state.json
COPY training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_model.zip \
     training_extension/frozen/ppo_parameters_12288_selected_20260726/selected_vecnormalize.pkl \
     training_extension/frozen/ppo_parameters_12288_selected_20260726/

# --- Level 02 trained model ---
COPY outputs/models/selected/best/best_model.zip outputs/models/selected/best/best_model.zip

EXPOSE 8000
CMD ["uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "8000"]
