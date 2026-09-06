#!/usr/bin/env bash
# Locks Jetson Orin Nano hardware clocks to peak performance:
# - GPU: 1020 MHz
# - EMC (Memory Bus): 3199 MHz (LPDDR5-6400)
# - CPU: 1728 MHz (all 6 cores)

echo "[+] Setting power mode to MAXN_SUPER (Mode 2)..."
sudo /usr/sbin/nvpmodel -m 2

echo "[+] Locking GPU clock to 1020 MHz..."
echo 1020000000 | sudo /usr/bin/tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/min_freq >/dev/null
echo 1020000000 | sudo /usr/bin/tee /sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu/max_freq >/dev/null

echo "[+] Locking EMC (Memory Bus) to peak 3199 MHz..."
echo 1 | sudo /usr/bin/tee /sys/kernel/debug/bpmp/debug/clk/emc/mrq_rate_locked >/dev/null
echo 1 | sudo /usr/bin/tee /sys/kernel/debug/bpmp/debug/bwmgr/bwmgr_halt >/dev/null
cat /sys/kernel/nvpmodel_clk_cap/emc | sudo /usr/bin/tee /sys/kernel/debug/bpmp/debug/clk/emc/rate >/dev/null

echo "[+] Locking CPU cores to 1728 MHz..."
for f in /sys/devices/system/cpu/cpu*/cpufreq/scaling_min_freq; do
    echo 1728000 | sudo /usr/bin/tee "$f" >/dev/null
done

echo "[+] Hardware clocks locked to peak frequencies successfully!"
