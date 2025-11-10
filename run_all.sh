#!/usr/bin/env bash
set -euo pipefail

LOGFILE="run_$(date +"%Y%m%d_%H%M%S").log"
exec > >(tee -a "$LOGFILE") 2>&1

# === パス設定（必要に応じて調整） ===
SIM_NAME="CUBEVis"
CONFIG_DIR="configs/${SIM_NAME}"                                  # 例: configs/CUBEVis
DATA_BASE="/home/tomoyam/Work/GitHub/ViSNeRF/data/${SIM_NAME}"    # 例: /home/.../ViSNeRF/data/CUBEVis

# === 1行=5要素: [0]=学習、[1]~[4]=eval ===
PAIRS=(
  # "mu0.0to1.0sp5_ts30to130sp10_view142 mu0.125to0.875sp2_ts30to130sp10_view1201_for_eval mu0.5to0.5sp1_ts30to130sp10_view_142_for_time_eval mu0.125to0.125sp1_ts30to130sp10_view142_for_sim_eval mu0.5to0.5sp1_ts30to130sp10_view1201_for_view_eval"
  # "mu0.0to1.0sp5_ts30to130sp10_view143 mu0.125to0.875sp2_ts30to130sp10_view1201_for_eval mu0.5to0.5sp1_ts30to130sp10_view_143_for_time_eval mu0.125to0.125sp1_ts30to130sp10_view143_for_sim_eval mu0.5to0.5sp1_ts30to130sp10_view1201_for_view_eval"
  "mu0.0to1.0sp5_ts30to130sp10_view181 mu0.125to0.875sp2_ts30to130sp10_view1201_for_eval mu0.5to0.5sp1_ts30to130sp10_view_181_for_time_eval mu0.125to0.125sp1_ts30to130sp10_view181_for_sim_eval mu0.5to0.5sp1_ts30to130sp10_view1201_for_view_eval"
  "mu0.0to1.0sp5_ts30to130sp10_view182 mu0.125to0.875sp2_ts30to130sp10_view1201_for_eval mu0.5to0.5sp1_ts30to130sp10_view_182_for_time_eval mu0.125to0.125sp1_ts30to130sp10_view182_for_sim_eval mu0.5to0.5sp1_ts30to130sp10_view1201_for_view_eval"
)

for row in "${PAIRS[@]}"; do
  # 5要素を取り出し
  read -r d0 d1 d2 d3 d4 <<< "$row"

  config_path="${CONFIG_DIR}/${d0}.txt"

  echo "======================================================="
  echo "[DATASET] ${d0}"
  echo "[CONFIG ] ${config_path}"
  echo "-------------------------------------------------------"

  # --- Config の存在チェック ---
  if [ ! -f "${config_path}" ]; then
    echo "ERROR: config not found -> ${config_path}" >&2
    exit 1
  fi

  # # --- Train（eval_datadirは指定しない） ---
  echo "Train: ${d0}"
  python train.py \
    --config "${config_path}"

  # --- Render相当（[1]~[4] を順に eval_datadir として実行）---
  for ev in "${d1}" "${d2}" "${d3}" "${d4}"; do
    if [ -n "${ev:-}" ] && [ "${ev}" != "None" ]; then
      eval_path="${DATA_BASE}/${ev}"
      echo "Eval: ${ev}"
      # ご指定に従い、render も train.py を --eval_datadir 付きで実行
      python train.py \
        --config "${config_path}" \
        --eval_datadir "${eval_path}"
    fi
  done

  # --- Metrics（1回/行）---
  echo "Metrics: ${d0}"
  python metrics.py \
    --config "${config_path}"

  echo "=== Done: ${d0} ==="
  echo
done
