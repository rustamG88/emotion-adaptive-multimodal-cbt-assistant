#!/bin/bash
#
# System Maintenance and Kernel Update Script for ALT Linux
# 
# This script performs:
# - Journal log cleanup
# - APT package updates and upgrades
# - Kernel updates
# - Old kernel removal
# - CUPS service maintenance
# - Automatic reboot scheduling if kernel was updated
#
# Usage: bash -lc 'path/to/system-maintenance.sh'
#

set -e

echo "[INFO] Starting system maintenance..."

# ============================================
# 1. Journal Log Cleanup
# ============================================
echo "[INFO] Cleaning journal logs..."
journalctl --rotate
journalctl --vacuum-time=1d
journalctl --vacuum-size=100M

# ============================================
# 2. APT Package Management
# ============================================
echo "[INFO] Cleaning APT cache..."
apt-get clean

echo "[INFO] Updating package lists..."
apt-get update

echo "[INFO] Performing distribution upgrade..."
DEBIAN_FRONTEND=noninteractive apt-get dist-upgrade -y

# ============================================
# 3. Kernel Update
# ============================================
echo "[INFO] Updating kernel..."
update-kernel -y || true

# ============================================
# 4. Kernel Version Check
# ============================================
echo "[INFO] Checking kernel versions..."
running="$(uname -r)"
latest="$(ls -1 /lib/modules 2>/dev/null | sort -V | tail -n1)"
need_reboot=0

echo "[INFO] Running kernel: $running"
echo "[INFO] Latest kernel:  $latest"

if [ -n "$latest" ] && [ "$latest" != "$running" ]; then
    need_reboot=1
    echo "[INFO] Kernel update detected - reboot will be scheduled"
else
    echo "[INFO] Kernel is up to date"
fi

# ============================================
# 5. Old Kernel Cleanup
# ============================================
echo "[INFO] Removing old kernels..."
remove-old-kernels -y || true

# ============================================
# 6. CUPS Service Maintenance
# ============================================
echo "[INFO] Performing CUPS maintenance..."

# Cancel all print jobs
if command -v cancel >/dev/null; then
    cancel -a || true
fi

# Restart CUPS service
systemctl try-restart cups.service || true

# Clear CUPS logs
: > /var/log/cups/access_log 2>/dev/null || true
: > /var/log/cups/error_log 2>/dev/null || true

# ============================================
# 7. Reboot Scheduling (if needed)
# ============================================
if [ "$need_reboot" -eq 1 ]; then
    echo "[WARNING] Kernel was updated - scheduling reboot..."
    
    # Try to schedule reboot at 18:35 using atd, fallback to shutdown in 5 minutes
    if systemctl is-active --quiet atd; then
        echo reboot | at 18:35
        echo "[INFO] Reboot scheduled at 18:35 via atd"
    else
        shutdown -r +5 "Kernel updated; rebooting in 5 minutes"
        echo "[INFO] Reboot scheduled in 5 minutes"
    fi
else
    echo "[INFO] No reboot needed"
fi

echo "[INFO] System maintenance completed successfully!"
