#!/bin/bash
# Monitoring stack: dcgm-exporter on every allocated node, Prometheus and Grafana once, on
# the node that runs the batch script. Needs lib.sh sourced first.
#
# Nothing here is a hardcoded host name: dcgm-exporter is started with `srun --overlap`,
# which inherits this job's allocation, and prometheus.yml is rendered from
# $SLURM_JOB_NODELIST on every run. sglang's own /metrics is scraped on
# $SGLANG_METRICS_PORTS (every serve command passes --enable-metrics).
#
# Inputs set by the caller before start_monitoring:
#   SGLANG_METRICS_PORTS         engine /metrics ports (single `sglang serve`, router workers or HTTP_PORT)
#   SGLANG_ROUTER_METRICS_PORT   optional, router fleet only

MONITOR_PIDS=()

monitoring_init() {
  # NOT the distroless dcgm-exporter :latest tag: it silently dies on newer drivers (starts,
  # one INFO line, exit 1, no usable error). The ubuntu-based tag has a shell and starts.
  DCGM_SIF=${DCGM_SIF:-$IMAGE_DIR/dcgm-exporter-ubuntu.sif}
  PROM_SIF=${PROM_SIF:-$IMAGE_DIR/prometheus.sif}
  GRAFANA_SIF=${GRAFANA_SIF:-$IMAGE_DIR/grafana.sif}
  # Datasource + dashboard provider config, and the two dashboard JSONs (sglang, dcgm).
  GRAFANA_PROVISIONING=${GRAFANA_PROVISIONING:-$SERVING_DIR/monitoring/grafana/provisioning}
  DASHBOARD_DIR=${DASHBOARD_DIR:-$SERVING_DIR/monitoring/dashboards}
  PROM_TEMPLATE=${PROM_TEMPLATE:-$SERVING_DIR/monitoring/prometheus.yml.tmpl}

  # Per-job dir: overlapping jobs each get their own TSDB and sqlite instead of fighting over
  # one storage lock. Nothing is removed on teardown except the rendered prometheus.yml, so
  # history survives, scoped per job id.
  MONITOR_RUN_DIR=$RUN_DIR/$SLURM_JOB_ID
  GRAFANA_DATA=$MONITOR_RUN_DIR/grafana-data
  PROM_DATA=$MONITOR_RUN_DIR/prometheus-data
  PROM_CONFIG_RUNTIME=$MONITOR_RUN_DIR/prometheus.runtime.yaml   # regenerated every run
  MONITOR_PIDS=()
}

# render_prometheus_config: write the prometheus.yml for this run to stdout.
render_prometheus_config() {
  local node p
  DCGM_TARGETS=""
  for node in $(list_nodes); do DCGM_TARGETS+="${DCGM_TARGETS:+ }${node}:${DCGM_PORT}"; done
  SGLANG_TARGETS=""
  for p in ${SGLANG_METRICS_PORTS:-}; do SGLANG_TARGETS+="${SGLANG_TARGETS:+ }localhost:${p}"; done
  LITELLM_PROM_SCRAPE=""
  if [ "${LITELLM_PROMETHEUS:-0}" = 1 ]; then LITELLM_PROM_SCRAPE=1; fi
  SGLANG_ROUTER_METRICS_PORT=${SGLANG_ROUTER_METRICS_PORT:-}
  render_template "$PROM_TEMPLATE"
}

start_monitoring() {
  local nodes host
  if [ "${ENABLE_MONITORING:-1}" != 1 ]; then log "[monitoring] disabled (ENABLE_MONITORING=$ENABLE_MONITORING)"; return 0; fi
  monitoring_init
  nodes=$(list_nodes | tr '\n' ' ')
  host=$(this_host)
  run mkdir -p "$MONITOR_RUN_DIR"

  log "[monitoring] job node(s): $nodes"
  log "[monitoring] prometheus + grafana landing on: $host"

  log "[monitoring] launching dcgm-exporter on all $SLURM_JOB_NUM_NODES node(s) ..."
  run_bg_log "$MONITOR_RUN_DIR/dcgm.log" \
    srun --overlap --ntasks-per-node=1 --nodes="$SLURM_JOB_NUM_NODES" \
    apptainer exec --nv "$DCGM_SIF" dcgm-exporter
  MONITOR_PIDS+=("$LAST_BG_PID")
  if ! is_dry; then sleep 5; fi   # let exporters bind before prometheus's first scrape

  # Rebuild prometheus.yml fresh each run: targets = THIS run's actual nodes.
  if is_dry; then
    log "[monitoring] prometheus.yml that would be written to $PROM_CONFIG_RUNTIME:"
    render_prometheus_config | sed 's/^/    | /'
  else
    render_prometheus_config >"$PROM_CONFIG_RUNTIME"
    log "[monitoring] wrote $PROM_CONFIG_RUNTIME"
  fi

  run mkdir -p "$PROM_DATA"
  run_bg_log "$MONITOR_RUN_DIR/prometheus.log" \
    apptainer run \
    --bind "$PROM_CONFIG_RUNTIME:/etc/prometheus/prometheus.yml" \
    --bind "$PROM_DATA:/prometheus" \
    "$PROM_SIF" \
    --config.file=/etc/prometheus/prometheus.yml \
    --storage.tsdb.path=/prometheus \
    --web.enable-lifecycle
  MONITOR_PIDS+=("$LAST_BG_PID")

  # Dashboard JSONs live in the same persistent volume grafana already reads
  # (/var/lib/grafana/dashboards, under $GRAFANA_DATA), copied in fresh each run so
  # editing or adding a .json under monitoring/dashboards is picked up by the next job.
  run mkdir -p "$GRAFANA_DATA/dashboards"
  if is_dry; then
    show_cmd "" cp "$DASHBOARD_DIR"/*.json "$GRAFANA_DATA/dashboards/"
  else
    cp "$DASHBOARD_DIR"/*.json "$GRAFANA_DATA/dashboards/" 2>/dev/null || true
  fi

  run_bg_log "$MONITOR_RUN_DIR/grafana.log" \
    apptainer run \
    --bind "$GRAFANA_DATA:/var/lib/grafana" \
    --bind "$GRAFANA_PROVISIONING/datasources:/etc/grafana/provisioning/datasources" \
    --bind "$GRAFANA_PROVISIONING/dashboards:/etc/grafana/provisioning/dashboards" \
    "$GRAFANA_SIF"
  MONITOR_PIDS+=("$LAST_BG_PID")

  log "[monitoring] up. From your laptop:"
  log "    ssh -L ${PROM_PORT}:localhost:${PROM_PORT} -L ${GRAFANA_PORT}:localhost:${GRAFANA_PORT} ${USER}@${host}${SSH_HOST_SUFFIX:-}"
  log "    (add -J ${USER}@<login-node> first if direct SSH to the compute node is not possible)"
  log "[monitoring] Grafana -> Dashboards -> 'SGLang Monitoring' folder has the SGLang and DCGM"
  log "    dashboards pre-loaded, the Prometheus datasource is provisioned too."
}

stop_monitoring() {
  if [ "${#MONITOR_PIDS[@]}" -eq 0 ]; then return 0; fi
  if is_dry; then return 0; fi
  log "[monitoring] shutting down ${#MONITOR_PIDS[@]} process(es) ..."
  kill "${MONITOR_PIDS[@]}" 2>/dev/null || true
  wait "${MONITOR_PIDS[@]}" 2>/dev/null || true
  rm -f "$PROM_CONFIG_RUNTIME"
}
