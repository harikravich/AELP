#!/usr/bin/env bash
set -euo pipefail

# Real-Time RL Training Monitor Dashboard
# Auto-refreshing comprehensive monitoring for GAELP RL agent training
#
# Features:
#   - Auto-refresh every 15 seconds
#   - Multiple training run tracking
#   - Progress bars and ETA calculation
#   - Performance metrics and trends
#   - Resource utilization monitoring
#   - Alert thresholds
#
# Usage:
#   bash AELP2/ops/training_status.sh                    # Auto-refresh mode
#   bash AELP2/ops/training_status.sh --once            # Single run
#   bash AELP2/ops/training_status.sh --refresh 30      # Custom refresh interval

PROJECT=${GOOGLE_CLOUD_PROJECT:-aura-thrive-platform}
DATASETS="gaelp_training,gaelp_sandbox_hari,gaelp_rnd_20250905"
WINDOW="120 MINUTE"  # lookback for trend
REFRESH_INTERVAL=15  # seconds
AUTO_REFRESH=true
SHOW_ALERTS=true

# Terminal colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --window) WINDOW="$2"; shift 2;;
    --refresh) REFRESH_INTERVAL="$2"; shift 2;;
    --once) AUTO_REFRESH=false; shift;;
    --no-alerts) SHOW_ALERTS=false; shift;;
    -h|--help)
      echo "Real-Time RL Training Monitor"
      echo "  --project   <id>            GCP project (default: $PROJECT)"
      echo "  --datasets  csv             Datasets to monitor (default: $DATASETS)"
      echo "  --window    'N MINUTE|HOUR' Trend window (default: $WINDOW)"
      echo "  --refresh   N               Refresh interval in seconds (default: $REFRESH_INTERVAL)"
      echo "  --once                      Run once without auto-refresh"
      echo "  --no-alerts                 Disable alert highlighting"
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 2;;
  esac
done

# Function to draw a progress bar
draw_progress_bar() {
  local current=$1
  local total=$2
  local width=${3:-40}
  local percent=$((current * 100 / total))
  local filled=$((width * current / total))
  
  printf "["
  printf "%0.s█" $(seq 1 $filled)
  printf "%0.s─" $(seq $((filled + 1)) $width)
  printf "] %3d%%" $percent
}

# Function to calculate ETA
calculate_eta() {
  local completed=$1
  local total=$2
  local elapsed_seconds=$3
  
  if [[ $completed -eq 0 ]]; then
    echo "Calculating..."
    return
  fi
  
  local rate=$(echo "scale=2; $completed / $elapsed_seconds" | bc)
  local remaining=$((total - completed))
  local eta_seconds=$(echo "scale=0; $remaining / $rate" | bc)
  
  if [[ $eta_seconds -lt 60 ]]; then
    echo "${eta_seconds}s"
  elif [[ $eta_seconds -lt 3600 ]]; then
    echo "$((eta_seconds / 60))m $((eta_seconds % 60))s"
  else
    echo "$((eta_seconds / 3600))h $((eta_seconds % 3600 / 60))m"
  fi
}

# Function to check alert conditions
check_alerts() {
  local metric=$1
  local value=$2
  local threshold=$3
  local comparison=$4  # "lt" or "gt"
  
  # Handle empty or null values
  if [[ -z "$value" ]] || [[ "$value" == "null" ]] || [[ "$value" == "" ]]; then
    echo -e "${YELLOW}--${NC}"
    return
  fi
  
  # Use awk instead of bc for better error handling
  if [[ "$comparison" == "lt" ]]; then
    result=$(awk -v v="$value" -v t="$threshold" 'BEGIN{print (v < t) ? 1 : 0}')
    if [[ "$result" == "1" ]]; then
      echo -e "${RED}⚠${NC}"
    else
      echo -e "${GREEN}✓${NC}"
    fi
  elif [[ "$comparison" == "gt" ]]; then
    result=$(awk -v v="$value" -v t="$threshold" 'BEGIN{print (v > t) ? 1 : 0}')
    if [[ "$result" == "1" ]]; then
      echo -e "${RED}⚠${NC}"
    else
      echo -e "${GREEN}✓${NC}"
    fi
  fi
}

# Main monitoring loop
monitor_training() {
  clear
  echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
  echo -e "${BOLD}${CYAN}║           GAELP RL TRAINING MONITOR - REAL-TIME DASHBOARD           ║${NC}"
  echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
  echo
  echo -e "${BOLD}Timestamp:${NC} $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo -e "${BOLD}Project:${NC} $PROJECT"
  echo -e "${BOLD}Refresh:${NC} Every ${REFRESH_INTERVAL}s | ${BOLD}Window:${NC} $WINDOW"
  echo
  
  IFS=',' read -r -a DS_ARR <<< "$DATASETS"
  
  for DS in "${DS_ARR[@]}"; do
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${BLUE}Dataset: $DS${NC}"
    echo -e "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Active Training Runs with Better ETA
    echo -e "\n${BOLD}${MAGENTA}▶ ACTIVE TRAINING RUNS & ETA${NC}"
    RUNS_DATA=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv --max_rows=10 \
      "SELECT run_id, status, episodes_requested, episodes_completed, 
              TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), start_time, SECOND) as elapsed_s,
              start_time,
              SAFE_DIVIDE(episodes_completed, TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), start_time, SECOND)) as eps_per_sec
       FROM \`$PROJECT.$DS.training_runs\`
       WHERE status IN ('RUNNING', 'PENDING', 'ACTIVE')
          OR start_time > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 HOUR)
       ORDER BY start_time DESC" 2>/dev/null | tail -n +2)
    
    if [[ -n "$RUNS_DATA" ]]; then
      while IFS=',' read -r run_id status requested completed elapsed start_time eps_per_sec; do
        requested=${requested:-0}
        completed=${completed:-0}
        elapsed=${elapsed:-0}
        eps_per_sec=${eps_per_sec:-0}
        
        echo -e "  ${BOLD}Run:${NC} $run_id"
        echo -e "  ${BOLD}Status:${NC} ${YELLOW}$status${NC} | Started: $start_time"
        printf "  ${BOLD}Progress:${NC} %'d/%'d episodes\n" "$completed" "$requested" 2>/dev/null || \
          echo -e "  ${BOLD}Progress:${NC} $completed/$requested episodes"
        echo -n "  "
        draw_progress_bar $completed $requested 30
        echo -e " | ${BOLD}ETA:${NC} $(calculate_eta $completed $requested $elapsed)"
        
        if [[ "$eps_per_sec" != "0" ]] && [[ "$eps_per_sec" != "" ]]; then
          eps_per_min=$(awk -v e="$eps_per_sec" 'BEGIN{printf "%.1f", e*60}')
          echo -e "  ${BOLD}Speed:${NC} $eps_per_min episodes/min"
        fi
        echo
      done <<< "$RUNS_DATA"
    else
      echo -e "  ${YELLOW}No active training runs${NC}"
    fi
    
    # Performance Metrics Summary
    echo -e "\n${BOLD}${MAGENTA}▶ PERFORMANCE METRICS (Last Hour)${NC}"
    METRICS=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv \
      "SELECT 
        IFNULL(ROUND(AVG(win_rate), 4), 0) as avg_win_rate,
        IFNULL(ROUND(AVG(roas), 2), 0) as avg_roas,
        IFNULL(ROUND(AVG(cac), 2), 0) as avg_cac,
        IFNULL(ROUND(MIN(epsilon), 4), 0) as min_epsilon,
        IFNULL(ROUND(MAX(epsilon), 4), 0) as max_epsilon,
        COUNT(*) as episode_count
       FROM \`$PROJECT.$DS.training_episodes\`
       WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)" 2>/dev/null | tail -n +2)
    
    if [[ -n "$METRICS" ]]; then
      IFS=',' read -r win_rate roas cac min_eps max_eps ep_count <<< "$METRICS"
      
      # Format values with defaults for empty
      win_rate=${win_rate:-"0"}
      roas=${roas:-"0"}  
      cac=${cac:-"0"}
      min_eps=${min_eps:-"0"}
      max_eps=${max_eps:-"0"}
      ep_count=${ep_count:-"0"}
      
      printf "  %-20s %-15s %s\n" "${BOLD}Win Rate:${NC}" "${win_rate}" "$(check_alerts "win_rate" "$win_rate" "0.10" "lt")"
      printf "  %-20s %-15s %s\n" "${BOLD}ROAS:${NC}" "${roas}" "$(check_alerts "roas" "$roas" "2.0" "lt")"
      printf "  %-20s %-15s %s\n" "${BOLD}CAC:${NC}" "\$${cac}" "$(check_alerts "cac" "$cac" "50" "gt")"
      printf "  %-20s %-15s\n" "${BOLD}Epsilon Range:${NC}" "${min_eps} - ${max_eps}"
      printf "  %-20s %-15s\n" "${BOLD}Episodes/Hour:${NC}" "${ep_count}"
    else
      echo -e "  ${YELLOW}No data available${NC}"
    fi
    
    # Training Velocity
    echo -e "\n${BOLD}${MAGENTA}▶ TRAINING VELOCITY${NC}"
    VELOCITY=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv \
      "WITH recent AS (
        SELECT 
          TIMESTAMP_TRUNC(timestamp, MINUTE) as minute,
          COUNT(*) as episodes_per_minute
        FROM \`$PROJECT.$DS.training_episodes\`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 MINUTE)
        GROUP BY minute
      )
      SELECT 
        IFNULL(ROUND(AVG(episodes_per_minute), 1), 0) as avg_epm,
        IFNULL(MAX(episodes_per_minute), 0) as max_epm,
        IFNULL(MIN(episodes_per_minute), 0) as min_epm
      FROM recent" 2>/dev/null | tail -n +2)
    
    if [[ -n "$VELOCITY" ]] && [[ "$VELOCITY" != ",," ]]; then
      IFS=',' read -r avg_epm max_epm min_epm <<< "$VELOCITY"
      avg_epm=${avg_epm:-"0"}
      max_epm=${max_epm:-"0"}
      min_epm=${min_epm:-"0"}
      
      printf "  %-20s %s\n" "${BOLD}Avg Episodes/Min:${NC}" "${avg_epm}"
      printf "  %-20s %s\n" "${BOLD}Max Episodes/Min:${NC}" "${max_epm}"
      printf "  %-20s %s\n" "${BOLD}Min Episodes/Min:${NC}" "${min_epm}"
    else
      echo -e "  ${YELLOW}No recent training activity${NC}"
    fi
    
    # RL LEARNING STATUS - What has the agent learned?
    echo -e "\n${BOLD}${CYAN}▶ RL AGENT LEARNING STATUS${NC}"
    LEARNING=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv --max_rows=1 \
      "WITH learning_progress AS (
        SELECT 
          COUNT(DISTINCT episode_id) as total_episodes,
          SUM(steps) as total_steps,
          SUM(auctions) as total_auctions,
          MIN(timestamp) as first_episode,
          MAX(timestamp) as last_episode,
          TIMESTAMP_DIFF(MAX(timestamp), MIN(timestamp), SECOND) as training_duration_s,
          
          -- Learning metrics over time windows
          AVG(CASE WHEN timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR) THEN win_rate END) as recent_win_rate,
          AVG(CASE WHEN timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR) 
                   AND timestamp <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 HOUR) THEN win_rate END) as early_win_rate,
          
          AVG(CASE WHEN timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR) THEN roas END) as recent_roas,
          AVG(CASE WHEN timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR) 
                   AND timestamp <= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 5 HOUR) THEN roas END) as early_roas,
          
          MIN(epsilon) as current_epsilon,
          MAX(epsilon) as initial_epsilon,
          
          COUNT(DISTINCT model_version) as model_versions
        FROM \`$PROJECT.$DS.training_episodes\`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
      )
      SELECT * FROM learning_progress" 2>/dev/null | tail -n +2)
    
    if [[ -n "$LEARNING" ]]; then
      IFS=',' read -r total_eps total_steps total_auctions first_ep last_ep duration_s recent_wr early_wr recent_roas early_roas curr_eps init_eps model_vers <<< "$LEARNING"
      
      # Set defaults for empty values
      total_eps=${total_eps:-0}
      total_steps=${total_steps:-0}
      total_auctions=${total_auctions:-0}
      recent_wr=${recent_wr:-0}
      early_wr=${early_wr:-0}
      recent_roas=${recent_roas:-0}
      early_roas=${early_roas:-0}
      curr_eps=${curr_eps:-0}
      init_eps=${init_eps:-1}
      model_vers=${model_vers:-0}
      duration_s=${duration_s:-0}
      
      # Calculate improvements
      wr_improvement=0
      roas_improvement=0
      if [[ "$early_wr" != "0" ]]; then
        wr_improvement=$(awk -v r="$recent_wr" -v e="$early_wr" 'BEGIN{printf "%.1f", ((r-e)/e*100)}')
      fi
      if [[ "$early_roas" != "0" ]]; then
        roas_improvement=$(awk -v r="$recent_roas" -v e="$early_roas" 'BEGIN{printf "%.1f", ((r-e)/e*100)}')
      fi
      
      echo -e "  ${BOLD}${GREEN}◆ Training Summary:${NC}"
      printf "  %-25s %'d\n" "Total Episodes:" "$total_eps" 2>/dev/null || printf "  %-25s %s\n" "Total Episodes:" "$total_eps"
      printf "  %-25s %'d\n" "Total Steps:" "$total_steps" 2>/dev/null || printf "  %-25s %s\n" "Total Steps:" "$total_steps"
      printf "  %-25s %'d\n" "Total Auctions:" "$total_auctions" 2>/dev/null || printf "  %-25s %s\n" "Total Auctions:" "$total_auctions"
      printf "  %-25s %s\n" "Model Versions:" "$model_vers"
      printf "  %-25s %s\n" "Training Duration:" "$((duration_s / 3600))h $((duration_s % 3600 / 60))m"
      
      echo -e "\n  ${BOLD}${GREEN}◆ Learning Progress:${NC}"
      printf "  %-25s %.4f → %.4f " "Win Rate Evolution:" "$early_wr" "$recent_wr"
      if (( $(awk -v i="$wr_improvement" 'BEGIN{print (i > 0) ? 1 : 0}') )); then
        echo -e "${GREEN}(+${wr_improvement}%)${NC}"
      else
        echo -e "${RED}(${wr_improvement}%)${NC}"
      fi
      
      printf "  %-25s %.2f → %.2f " "ROAS Evolution:" "$early_roas" "$recent_roas"
      if (( $(awk -v i="$roas_improvement" 'BEGIN{print (i > 0) ? 1 : 0}') )); then
        echo -e "${GREEN}(+${roas_improvement}%)${NC}"
      else
        echo -e "${RED}(${roas_improvement}%)${NC}"
      fi
      
      printf "  %-25s %.4f → %.4f\n" "Epsilon Decay:" "$init_eps" "$curr_eps"
      
      # Training quality indicator
      echo -e "\n  ${BOLD}${GREEN}◆ Training Direction:${NC}"
      if (( $(awk -v i="$wr_improvement" 'BEGIN{print (i > 5) ? 1 : 0}') )); then
        echo -e "  ${GREEN}✓ Win Rate: IMPROVING WELL${NC}"
      elif (( $(awk -v i="$wr_improvement" 'BEGIN{print (i > 0) ? 1 : 0}') )); then
        echo -e "  ${YELLOW}↑ Win Rate: SLOWLY IMPROVING${NC}"
      else
        echo -e "  ${RED}✗ Win Rate: NEEDS ATTENTION${NC}"
      fi
      
      if (( $(awk -v i="$roas_improvement" 'BEGIN{print (i > 5) ? 1 : 0}') )); then
        echo -e "  ${GREEN}✓ ROAS: IMPROVING WELL${NC}"
      elif (( $(awk -v i="$roas_improvement" 'BEGIN{print (i > 0) ? 1 : 0}') )); then
        echo -e "  ${YELLOW}↑ ROAS: SLOWLY IMPROVING${NC}"
      else
        echo -e "  ${RED}✗ ROAS: NEEDS ATTENTION${NC}"
      fi
    else
      echo -e "  ${YELLOW}No learning data in last 24 hours${NC}"
    fi
    
    # CONVERGENCE METRICS
    echo -e "\n${BOLD}${CYAN}▶ CONVERGENCE & STABILITY${NC}"
    CONVERGENCE=$(bq --project_id="$PROJECT" query --use_legacy_sql=false --format=csv \
      "WITH hourly_metrics AS (
        SELECT 
          TIMESTAMP_TRUNC(timestamp, HOUR) as hour,
          AVG(win_rate) as wr_avg,
          STDDEV(win_rate) as wr_std,
          AVG(roas) as roas_avg,
          STDDEV(roas) as roas_std,
          COUNT(*) as episodes
        FROM \`$PROJECT.$DS.training_episodes\`
        WHERE timestamp > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 6 HOUR)
        GROUP BY hour
        HAVING episodes > 10
      )
      SELECT 
        ROUND(AVG(wr_std), 5) as avg_wr_std,
        ROUND(STDDEV(wr_avg), 5) as wr_volatility,
        ROUND(AVG(roas_std), 4) as avg_roas_std,
        ROUND(STDDEV(roas_avg), 4) as roas_volatility
      FROM hourly_metrics" 2>/dev/null | tail -n +2)
    
    if [[ -n "$CONVERGENCE" ]]; then
      IFS=',' read -r wr_std wr_vol roas_std roas_vol <<< "$CONVERGENCE"
      
      wr_std=${wr_std:-0}
      wr_vol=${wr_vol:-0}
      roas_std=${roas_std:-0}
      roas_vol=${roas_vol:-0}
      
      printf "  %-25s %s " "Win Rate Volatility:" "$wr_vol"
      if (( $(awk -v v="$wr_vol" 'BEGIN{print (v < 0.01) ? 1 : 0}') )); then
        echo -e "${GREEN}[STABLE]${NC}"
      elif (( $(awk -v v="$wr_vol" 'BEGIN{print (v < 0.05) ? 1 : 0}') )); then
        echo -e "${YELLOW}[CONVERGING]${NC}"
      else
        echo -e "${RED}[UNSTABLE]${NC}"
      fi
      
      printf "  %-25s %s " "ROAS Volatility:" "$roas_vol"
      if (( $(awk -v v="$roas_vol" 'BEGIN{print (v < 0.5) ? 1 : 0}') )); then
        echo -e "${GREEN}[STABLE]${NC}"
      elif (( $(awk -v v="$roas_vol" 'BEGIN{print (v < 2.0) ? 1 : 0}') )); then
        echo -e "${YELLOW}[CONVERGING]${NC}"
      else
        echo -e "${RED}[UNSTABLE]${NC}"
      fi
    fi
    
    # Latest Episode Details
    echo -e "\n${BOLD}${MAGENTA}▶ LATEST EPISODES${NC}"
    bq --project_id="$PROJECT" query --use_legacy_sql=false --format=pretty \
      "SELECT 
        FORMAT_TIMESTAMP('%H:%M:%S', timestamp) as time,
        steps, 
        ROUND(win_rate,3) AS win_rate,
        ROUND(spend,2) AS spend, 
        ROUND(revenue,2) AS revenue,
        ROUND(roas,2) AS roas
       FROM \`$PROJECT.$DS.training_episodes\`
       ORDER BY timestamp DESC 
       LIMIT 3" 2>/dev/null || true
    
    # System Resources
    echo -e "\n${BOLD}${MAGENTA}▶ SYSTEM RESOURCES${NC}"
    PROC_COUNT=$(ps -ef | grep -iE "production_orchestrator|run_aelp2|run_quick_fidelity" | grep -v grep | wc -l)
    echo -e "  ${BOLD}Active Processes:${NC} $PROC_COUNT"
    
    if [[ $PROC_COUNT -gt 0 ]]; then
      ps -eo pid,pcpu,pmem,etime,cmd | grep -iE "production_orchestrator|run_aelp2|run_quick_fidelity" | grep -v grep | head -3 | while read line; do
        echo "  $line" | cut -c1-80
      done
    fi
  done
  
  echo -e "\n${BOLD}${CYAN}────────────────────────────────────────────────────────────────────────${NC}"
  
  if [[ "$AUTO_REFRESH" == true ]]; then
    echo -e "Next refresh in ${REFRESH_INTERVAL}s | Press Ctrl+C to exit"
  fi
}

# Main execution
if [[ "$AUTO_REFRESH" == true ]]; then
  trap "echo -e '\n${BOLD}Monitoring stopped.${NC}'; exit 0" INT
  
  while true; do
    monitor_training
    sleep $REFRESH_INTERVAL
  done
else
  monitor_training
fi

