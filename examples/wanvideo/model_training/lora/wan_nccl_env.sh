# shellcheck shell=bash
# NCCL env for Wan multi-GPU on GCP (4× RTX PRO 6000, local NVMe under /home/tungdo/storage).
# Sourced by Wan2.2-Animate-14B-meanflow.sh and train.sh before accelerate launch.
#
# Prior working host settings (bash history): NCCL_IB_DISABLE=1; optional P2P/SHM off.
# GCP image sets NCCL_NET=gIB — conflicts with TORCH_NCCL_ASYNC_ERROR_HANDLING=1 (PyTorch may set it).
#
# WAN_NCCL_MODE:
#   none_net  env -u NCCL_NET; NCCL_NET=none; IB/P2P/SHM off (bash_history wan_train_2gpu_nccl_gib_off.log)
#   legacy    NCCL_IB_DISABLE=1 + TORCH_NCCL_ASYNC_ERROR_HANDLING=1 (your prior bash_history on this VM)
#   gcp       unset TORCH_NCCL_ASYNC_ERROR_HANDLING; NCCL_NET=Socket; NCCL_IB_DISABLE=1
#   gcp_safe  gcp + NCCL_P2P_DISABLE=1 + NCCL_SHM_DISABLE=1
#   gib_only  use image NCCL_NET=gIB only; unset TORCH_NCCL_ASYNC_ERROR_HANDLING (recommended by nccl-shim WARN)
#   local     single-node: drop gIB libnccl-net from LD_LIBRARY_PATH (stock NCCL + GPU P2P)
#   none      do not change NCCL (image defaults: NCCL_NET=gIB)

wan_nccl_configure() {
  local num_gpus="${1:-1}"
  [[ "$num_gpus" == "1" ]] && return 0

  local mode="${WAN_NCCL_MODE:-gib_only}"
  case "$mode" in
    none) return 0 ;;
    gib_only)
      unset TORCH_NCCL_ASYNC_ERROR_HANDLING WAN_NCCL_MODE
      # Shim rejects user overrides of IB disable / socket ifname, but *requires* image tuning vars
      # (NCCL_IB_TC=52, NCCL_IB_FIFO_TC=84, …). Do not unset those.
      unset NCCL_IB_DISABLE NCCL_IBEXT_DISABLE NCCL_P2P_DISABLE NCCL_SHM_DISABLE NCCL_SOCKET_IFNAME
      export NCCL_NET="${NCCL_NET:-gIB}"
      export NCCL_IB_TC="${NCCL_IB_TC:-52}"
      export NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-84}"
      export NCCL_IB_ADAPTIVE_ROUTING="${NCCL_IB_ADAPTIVE_ROUTING:-1}"
      export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
      export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
      echo "NCCL (gib_only): NCCL_NET=${NCCL_NET} NCCL_IB_TC=${NCCL_IB_TC} (image gIB tuning preserved)"
      return 0
      ;;
    none_net)
      # Single-node fallback: P2P/NVLink only (no gIB). Keep image IB tuning vars for shim.
      unset TORCH_NCCL_ASYNC_ERROR_HANDLING WAN_NCCL_MODE
      unset NCCL_NET NCCL_IB_DISABLE NCCL_IBEXT_DISABLE NCCL_SOCKET_IFNAME
      export NCCL_NET=none
      export NCCL_IB_TC="${NCCL_IB_TC:-52}"
      export NCCL_IB_FIFO_TC="${NCCL_IB_FIFO_TC:-84}"
      export NCCL_IB_ADAPTIVE_ROUTING="${NCCL_IB_ADAPTIVE_ROUTING:-1}"
      export NCCL_IB_QPS_PER_CONNECTION="${NCCL_IB_QPS_PER_CONNECTION:-4}"
      export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
      echo "NCCL (none_net): NCCL_NET=none (P2P only; image IB_TC preserved)"
      return 0
      ;;
    legacy)
      export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
      export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
      export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
      echo "NCCL (legacy): NCCL_IB_DISABLE=${NCCL_IB_DISABLE} TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING} NCCL_NET=${NCCL_NET:-<image default>}"
      return 0
      ;;
    local)
      unset TORCH_NCCL_ASYNC_ERROR_HANDLING WAN_NCCL_MODE
      unset NCCL_NET NCCL_TUNER_CONFIG_PATH NCCL_IB_DISABLE NCCL_IBEXT_DISABLE NCCL_SOCKET_IFNAME
      if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
        LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v '/usr/local/gib' | paste -sd: -)"
        export LD_LIBRARY_PATH
      fi
      export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
      echo "NCCL (local): stock NCCL (gIB plugin stripped from LD_LIBRARY_PATH) for single-node P2P"
      return 0
      ;;
    gcp|gcp_safe) ;;
    *)
      echo "wan_nccl_configure: unknown WAN_NCCL_MODE=$mode (use legacy, gcp, gcp_safe, socket, or none)" >&2
      return 1
      ;;
  esac

  unset TORCH_NCCL_ASYNC_ERROR_HANDLING
  unset NCCL_NET
  export NCCL_NET="${NCCL_NET:-Socket}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_IBEXT_DISABLE="${NCCL_IBEXT_DISABLE:-1}"

  if [[ "$mode" == "gcp_safe" ]]; then
    export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
    export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
  fi

  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  unset WAN_NCCL_MODE
  echo "NCCL (${mode}): NCCL_NET=${NCCL_NET} NCCL_IB_DISABLE=${NCCL_IB_DISABLE} NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0} NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-0} (TORCH_NCCL_ASYNC_ERROR_HANDLING unset)"
}
